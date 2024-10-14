/*
 *      uvcvideo.c  --  USB Video Class driver
 *
 *      Copyright (C) 2005-2006
 *          Laurent Pinchart (laurent.pinchart@skynet.be)
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License as published by
 *      the Free Software Foundation; either version 2 of the License, or
 *      (at your option) any later version.
 *
 */

/*
 * WARNING: This driver is definitely *NOT* complete. It will (hopefully)
 * support UVC devices with a camera sensors, a processing unit and several
 * optional extension units. Single-input selector units are ignored.
 * Everything else is unsupported.
 *
 * The driver doesn't support the deprecated v4l1 interface. It implements the
 * mmap capture method only, and doesn't do any image format conversion in
 * software. If your user-space application doesn't support YUYV or MJPEG, fix
 * it :-). Please note that the MJPEG data have been stripped from their
 * Huffman tables (DHT marker), you will need to add it back if your JPEG
 * codec can't handle MJPEG data.
 *
 * Although the driver compiles on 2.6.12, you should use a 2.6.15 or newer
 * kernel because of USB issues.
 */
 
#include <linux/kernel.h>
#include <linux/version.h>
#include <linux/list.h>
#include <linux/module.h>
#include <linux/usb.h>
#include <linux/videodev.h>
#include <linux/vmalloc.h>
#include <linux/wait.h>
#include <asm/atomic.h>

#if LINUX_VERSION_CODE > KERNEL_VERSION(2,6,15)
#include <media/v4l2-common.h>
#endif

#include "uvcvideo.h"

#define DRIVER_AUTHOR		"Laurent Pinchart <laurent.pinchart@skynet.be>"
#define DRIVER_DESC		"USB Video Class driver"
#define DRIVER_VERSION		"0.1.0"

/* ------------------------------------------------------------------------
 * Control, formats, ...
 */

static struct uvc_format_desc uvc_fmts[] = {
	{
		.guid		= UVC_GUID_FORMAT_YUY2,
		.fcc		= V4L2_PIX_FMT_YUYV,
	},
	{
		.guid		= UVC_GUID_FORMAT_NV12,
		.fcc		= V4L2_PIX_FMT_NV12,
	},
};

#if 0
static void uvc_print_streaming_control(struct uvc_streaming_control *ctrl)
{
	printk(KERN_DEBUG "bmHint:                      0x%04x\n", ctrl->bmHint);
	printk(KERN_DEBUG "bFormatIndex:                   %3u\n", ctrl->bFormatIndex);
	printk(KERN_DEBUG "bFrameIndex:                    %3u\n", ctrl->bFrameIndex);
	printk(KERN_DEBUG "dwFrameInterval:          %9u\n", ctrl->dwFrameInterval);
	printk(KERN_DEBUG "wKeyFrameRate:                %5u\n", ctrl->wKeyFrameRate);
	printk(KERN_DEBUG "wPFrameRate:                  %5u\n", ctrl->wPFrameRate);
	printk(KERN_DEBUG "wCompQuality:                 %5u\n", ctrl->wCompQuality);
	printk(KERN_DEBUG "wCompWindowSize:              %5u\n", ctrl->wCompWindowSize);
	printk(KERN_DEBUG "wDelay:                       %5u\n", ctrl->wDelay);
	printk(KERN_DEBUG "dwMaxVideoFrameSize:      %9u\n", ctrl->dwMaxVideoFrameSize);
	printk(KERN_DEBUG "dwMaxPayloadTransferSize: %9u\n", ctrl->dwMaxPayloadTransferSize);
	printk(KERN_DEBUG "dwClockFrequency:         %9u\n", ctrl->dwClockFrequency);
	printk(KERN_DEBUG "bmFramingInfo:                 0x%02x\n", ctrl->bmFramingInfo);
	printk(KERN_DEBUG "bPreferedVersion:               %3u\n", ctrl->bPreferedVersion);
	printk(KERN_DEBUG "bMinVersion:                    %3u\n", ctrl->bMinVersion);
	printk(KERN_DEBUG "bMaxVersion:                    %3u\n", ctrl->bMaxVersion);
}
#endif

/* ------------------------------------------------------------------------
 * Utility functions
 */

struct usb_host_endpoint *uvc_find_endpoint(struct usb_host_interface *alts,
		__u8 epaddr)
{
	struct usb_host_endpoint *ep;
	unsigned int i;

	for (i = 0; i < alts->desc.bNumEndpoints; ++i) {
		ep = &alts->endpoint[i];
		if (ep->desc.bEndpointAddress == epaddr)
			return ep;
	}

	return NULL;
}

static __u32 uvc_guid_to_fcc(const __u8 guid[16])
{
	unsigned int len = ARRAY_SIZE(uvc_fmts);
	unsigned int i;

	for (i = 0; i < len; ++i) {
		if (memcmp(guid, uvc_fmts[i].guid, 16) == 0)
			return uvc_fmts[i].fcc;
	}

	return -1;
}

static __u32 uvc_colorspace(const __u8 primaries)
{
	static const __u8 colorprimaries[] = {
		0,
		V4L2_COLORSPACE_SRGB,
		V4L2_COLORSPACE_470_SYSTEM_M,
		V4L2_COLORSPACE_470_SYSTEM_BG,
		V4L2_COLORSPACE_SMPTE170M,
		V4L2_COLORSPACE_SMPTE240M,
	};

	if (primaries < ARRAY_SIZE(colorprimaries))
		return colorprimaries[primaries];

	return 0;
}

/* Simplify a fraction using a simple continued fraction decomposition. The
 * idea here is to convert fractions such as 333333/10000000 to 1/30 using
 * 32 bit arithmetic only. The algorithm is not perfect and relies upon two
 * arbitrary parameters to remove non-significative terms from the simple
 * continued fraction decomposition. Using 8 and 333 for n_terms and threshold
 * respectively seems to give nice results.
 */
void uvc_simplify_fraction(uint32_t *numerator, uint32_t *denominator,
		unsigned int n_terms, unsigned int threshold)
{
	uint32_t *an;
	uint32_t x, y, r;
	unsigned int i, n;

	an = kmalloc(n_terms * sizeof *an, GFP_KERNEL);
	if (an == NULL)
		return;

	/* Convert the fraction to a simple continued fraction. See
	 * http://mathforum.org/dr.math/faq/faq.fractions.html
	 * Stop if the current term is bigger than or equal to the given
	 * threshold.
	 */
	x = *numerator;
	y = *denominator;

	for (n = 0; n < n_terms && y != 0; ++n) {
		an[n] = x / y;
		if (an[n] >= threshold) {
			if (n < 2)
				n++;
			break;
		}

		r = x - an[n] * y;
		x = y;
		y = r;
	}

	/* Expand the simple continued fraction back to an integer fraction. */
	x = 0;
	y = 1;

	for (i = n; i > 0; --i) {
		r = y;
		y = an[i-1] * y + x;
		x = r;
	}

	*numerator = y;
	*denominator = x;
	kfree(an);
}

/* Convert a fraction to a frame interval in 100ns multiples. The idea here is
 * to compute numerator / denominator * 10000000 using 32 bit fixed point
 * arithmetic only.
 */
uint32_t uvc_fraction_to_interval(uint32_t numerator, uint32_t denominator)
{
	uint32_t multiplier;

	/* Saturate the result if the operation would overflow. */
	if (denominator == 0 ||
	    numerator/denominator >= ((uint32_t)-1)/10000000)
		return (uint32_t)-1;

	/* Divide both the denominator and the multiplier by two until
	 * numerator * multiplier doesn't overflow. If anyone knows a better
	 * algorithm please let me know.
	 */
	multiplier = 10000000;
	while (numerator > ((uint32_t)-1)/multiplier) {
		multiplier /= 2;
		denominator /= 2;
	}

	return denominator ? numerator * multiplier / denominator : 0;
}

/* ------------------------------------------------------------------------
 * Terminal and unit management
 */

static struct uvc_entity *uvc_entity_by_id(struct uvc_device *dev, int id)
{
	struct uvc_entity *entity;

	list_for_each_entry(entity, &dev->entities, list) {
		if (entity->id == id)
			return entity;
	}

	return NULL;
}

static struct uvc_entity *uvc_entity_by_reference(struct uvc_device *dev,
	int id, struct uvc_entity *entity)
{
	unsigned int i;

	if (entity == NULL)
		entity = list_entry(&dev->entities, struct uvc_entity, list);

	list_for_each_entry_continue(entity, &dev->entities, list) {
		switch (entity->type) {
		case TT_STREAMING:
			if (entity->output.bSourceID == id)
				return entity;
			break;

		case VC_PROCESSING_UNIT:
			if (entity->processing.bSourceID == id)
				return entity;
			break;

		case VC_SELECTOR_UNIT:
			for (i = 0; i < entity->selector.bNrInPins; ++i)
				if (entity->selector.baSourceID[i] == id)
					return entity;
			break;

		case VC_EXTENSION_UNIT:
			for (i = 0; i < entity->extension.bNrInPins; ++i)
				if (entity->extension.baSourceID[i] == id)
					return entity;
			break;
		}
	}

	return NULL;
}

/* ------------------------------------------------------------------------
 * Descriptors handling
 */

static int uvc_parse_format(struct uvc_device *dev,
	struct uvc_streaming *streaming, struct uvc_format *format,
	__u32 **intervals, unsigned char *buffer, int buflen)
{
	struct usb_interface *intf = streaming->intf;
	struct usb_host_interface *alts = intf->cur_altsetting;
	struct uvc_frame *frame;
	const unsigned char *start = buffer;
	unsigned int interval;
	unsigned int i, n;
	__u8 ftype;

	format->type = buffer[2];
	format->index = buffer[3];

	switch (buffer[2]) {
	case VS_FORMAT_UNCOMPRESSED:
		if (buflen < 27) {
			uvc_trace(UVC_TRACE_DESCR, "device %d videostreaming"
			       "interface %d FORMAT error\n",
			       dev->udev->devnum,
			       alts->desc.bInterfaceNumber);
			return -EINVAL;
		}

		strncpy(format->name, "Uncompressed", sizeof format->name);
		format->fcc = uvc_guid_to_fcc(&buffer[5]);
		format->flags = 0;
		format->bpp = 16;
		ftype = VS_FRAME_UNCOMPRESSED;
		break;

	case VS_FORMAT_MJPEG:
		if (buflen < 11) {
			uvc_trace(UVC_TRACE_DESCR, "device %d videostreaming"
			       "interface %d FORMAT error\n",
			       dev->udev->devnum,
			       alts->desc.bInterfaceNumber);
			return -EINVAL;
		}

		strncpy(format->name, "MJPEG", sizeof format->name);
		format->fcc = V4L2_PIX_FMT_MJPEG;
		format->flags = V4L2_FMT_FLAG_COMPRESSED;
		format->bpp = 0;
		ftype = VS_FRAME_MJPEG;
		break;

	case VS_FORMAT_MPEG2TS:
	case VS_FORMAT_DV:
	case VS_FORMAT_FRAME_BASED:
	case VS_FORMAT_STREAM_BASED:
		/* Not supported yet. */
	default:
		uvc_trace(UVC_TRACE_DESCR, "device %d videostreaming"
		       "interface %d unsupported format %u\n",
		       dev->udev->devnum, alts->desc.bInterfaceNumber,
		       buffer[2]);
		return -EINVAL;
	}

	uvc_trace(UVC_TRACE_DESCR, "Found format %s.\n", format->name);

	buflen -= buffer[0];
	buffer += buffer[0];

	while (buflen > 2 && buffer[2] == ftype) {
		frame = &format->frame[format->nframes];

		switch (buffer[2]) {
		case VS_FRAME_UNCOMPRESSED:
		case VS_FRAME_MJPEG:
			n = buflen > 25 ? buffer[25] : 0;
			n = n ? n : 3;

			if (buflen < 26 + 4*n) {
				uvc_trace(UVC_TRACE_DESCR, "device %d videostreaming"
				       "interface %d FRAME error\n",
				       dev->udev->devnum,
				       alts->desc.bInterfaceNumber);
				return -EINVAL;
			}

			frame->bFrameIndex = buffer[3];
			frame->bmCapabilities = buffer[4];
			frame->wWidth = le16_to_cpup((__le16*)&buffer[5]);
			frame->wHeight = le16_to_cpup((__le16*)&buffer[7]);
			frame->dwMinBitRate = le32_to_cpup((__le32*)&buffer[9]);
			frame->dwMaxBitRate = le32_to_cpup((__le32*)&buffer[13]);
			frame->dwMaxVideoFrameBufferSize = le32_to_cpup((__le32*)&buffer[17]);
			frame->bFrameIntervalType = buffer[25];
			frame->dwFrameInterval = *intervals;

			/* Some bogus devices report dwMinFrameInterval equal
			 * to dwMaxFrameInterval and have dwFrameIntervalStep
			 * set to zero. Setting all null intervals to 1 fixes
			 * the problem and some other divisions by zero which
			 * could happen.
			 */
			for (i = 0; i < n; ++i) {
				interval = le32_to_cpup((__le32*)&buffer[26+4*i]);
				*(*intervals)++ = interval ? interval : 1;
			}

			/* Make sure that the default frame interval stays
			 * between the boundaries.
			 */
			n -= frame->bFrameIntervalType ? 1 : 2;
			interval = le32_to_cpup((__le32*)&buffer[21]);
			if (interval < frame->dwFrameInterval[0])
				interval = frame->dwFrameInterval[0];
			else if (interval > frame->dwFrameInterval[n])
				interval = frame->dwFrameInterval[n];

			frame->dwDefaultFrameInterval = interval;

			uvc_trace(UVC_TRACE_DESCR, "- %ux%u (%u.%u fps)\n",
				frame->wWidth, frame->wHeight,
				10000000/frame->dwDefaultFrameInterval,
				(100000000/frame->dwDefaultFrameInterval)%10);

			break;
		}

		format->nframes++;
		buflen -= buffer[0];
		buffer += buffer[0];
	}

	if (buflen > 2 && buffer[2] == VS_STILL_IMAGE_FRAME) {
		buflen -= buffer[0];
		buffer += buffer[0];
	}

	if (buflen > 2 && buffer[2] == VS_COLORFORMAT) {
		if (buflen < 6) {
			uvc_trace(UVC_TRACE_DESCR, "device %d videostreaming"
			       "interface %d COLORFORMAT error\n",
			       dev->udev->devnum,
			       alts->desc.bInterfaceNumber);
			return -EINVAL;
		}

		format->colorspace = uvc_colorspace(buffer[3]);

		buflen -= buffer[0];
		buffer += buffer[0];
	}

	return buffer - start;
}

static int uvc_parse_streaming(struct uvc_device *dev,
	struct uvc_streaming *streaming)
{
	struct uvc_format *format;
	struct uvc_frame *frame;
	struct usb_interface *intf = streaming->intf;
	struct usb_host_interface *alts = &intf->altsetting[0];
	unsigned char *_buffer, *buffer = alts->extra;
	int _buflen, buflen = alts->extralen;
	unsigned int nformats = 0, nframes = 0, nintervals = 0;
	unsigned int size, i, n, p;
	__u32 *interval;
	__u16 psize;
	int ret;

	/* The Pico iMage webcam has its class-specific interface descriptors
	 * after the endpoint descriptors.
	 */
	if (buflen == 0) {
		for (i = 0; i < alts->desc.bNumEndpoints; ++i) {
			struct usb_host_endpoint *ep = &alts->endpoint[i];

			if (ep->extralen == 0)
				continue;

			if (ep->extralen > 2 && ep->extra[1] == USB_DT_CS_INTERFACE) {
				uvc_trace(UVC_TRACE_DESCR, "trying extra data "
					"from endpoint %u.\n", i);
				buffer = alts->endpoint[i].extra;
				buflen = alts->endpoint[i].extralen;
				break;
			}
		}
	}

	/* Skip the standard interface descriptors. */
	while (buflen > 2 && buffer[1] != USB_DT_CS_INTERFACE) {
		buflen -= buffer[0];
		buffer += buffer[0];
	}

	if (buflen <= 2) {
		uvc_trace(UVC_TRACE_DESCR, "no class-specific streaming "
			"interface descriptors found.\n");
		return -EINVAL;
	}

	/* Parse the header descriptor. */
	if (buffer[2] == VS_OUTPUT_HEADER) {
		uvc_trace(UVC_TRACE_DESCR, "device %d videostreaming interface "
			"%d OUTPUT HEADER descriptor is not supported.\n",
			dev->udev->devnum, alts->desc.bInterfaceNumber);
		return -EINVAL;
	}
	else if (buffer[2] == VS_INPUT_HEADER) {
		p = buflen >= 5 ? buffer[3] : 0;
		n = buflen >= 12 ? buffer[12] : 0;

		if (buflen < 13 + p*n || buffer[2] != VS_INPUT_HEADER) {
			uvc_trace(UVC_TRACE_DESCR, "device %d videostreaming "
				"interface %d INPUT HEADER descriptor is "
				"invalid.\n", dev->udev->devnum,
				alts->desc.bInterfaceNumber);
			return -EINVAL;
		}

		streaming->input.bNumFormats = p;
		streaming->input.bEndpointAddress = buffer[6];
		streaming->input.bmInfo = buffer[7];
		streaming->input.bTerminalLink = buffer[8];
		streaming->input.bStillCaptureMethod = buffer[9];
		streaming->input.bTriggerSupport = buffer[10];
		streaming->input.bTriggerUsage = buffer[11];
		streaming->input.bControlSize = n;

		streaming->input.bmaControls = kmalloc(p*n, GFP_KERNEL);
		if (streaming->input.bmaControls == NULL)
			return -ENOMEM;

		memcpy(streaming->input.bmaControls, &buffer[13], p*n);
	}
	else {
		uvc_trace(UVC_TRACE_DESCR, "device %d videostreaming interface "
			"%d HEADER descriptor not found.\n", dev->udev->devnum,
			alts->desc.bInterfaceNumber);
		return -EINVAL;
	}

	buflen -= buffer[0];
	buffer += buffer[0];

	_buffer = buffer;
	_buflen = buflen;

	/* Count the format and frame descriptors. */
	while (_buflen > 2) {
		switch (_buffer[2]) {
		case VS_FORMAT_UNCOMPRESSED:
		case VS_FORMAT_MJPEG:
			nformats++;
			break;

		case VS_FORMAT_MPEG2TS:
		case VS_FORMAT_DV:
		case VS_FORMAT_FRAME_BASED:
		case VS_FORMAT_STREAM_BASED:
			uvc_trace(UVC_TRACE_DESCR, "device %d videostreaming "
				"interface %d FORMAT %u is not supported.\n",
				dev->udev->devnum,
				alts->desc.bInterfaceNumber, _buffer[2]);
			break;

		case VS_FRAME_UNCOMPRESSED:
		case VS_FRAME_MJPEG:
			nframes++;
			if (_buflen > 25)
				nintervals += _buffer[25] ? _buffer[25] : 3;
			break;

		case VS_FRAME_FRAME_BASED:
			nframes++;
			if (_buflen > 21)
				nintervals += _buffer[21] ? _buffer[21] : 3;
			break;
		}

		_buflen -= _buffer[0];
		_buffer += _buffer[0];
	}

	if (nformats == 0) {
		uvc_trace(UVC_TRACE_DESCR, "device %d videostreaming interface "
			"%d has no supported formats defined.\n",
			dev->udev->devnum, alts->desc.bInterfaceNumber);
		return -EINVAL;
	}

	size = nformats * sizeof *format + nframes * sizeof *frame
	     + nintervals * sizeof *interval;
	format = kzalloc(size, GFP_KERNEL);
	if (format == NULL)
		return -ENOMEM;

	frame = (struct uvc_frame*)&format[nformats];
	interval = (__u32*)&frame[nframes];

	streaming->format = format;
	streaming->nformats = nformats;

	/* Parse the format descriptors. */
	while (buflen > 2) {
		switch (buffer[2]) {
		case VS_FORMAT_UNCOMPRESSED:
		case VS_FORMAT_MJPEG:
			format->frame = frame;
			ret = uvc_parse_format(dev, streaming, format,
				&interval, buffer, buflen);
			if (ret < 0)
				return ret;

			frame += format->nframes;
			format++;

			buflen -= ret;
			buffer += ret;
			continue;

		default:
			break;
		}

		buflen -= buffer[0];
		buffer += buffer[0];
	}

	/* Parse the alternate settings to find the maximum bandwidth. */
	for (i = 0; i < intf->num_altsetting; ++i) {
		struct usb_host_endpoint *ep;
		alts = &intf->altsetting[i];
		ep = uvc_find_endpoint(alts,
				streaming->input.bEndpointAddress);
		if (ep == NULL)
			continue;

		psize = le16_to_cpu(ep->desc.wMaxPacketSize);
		psize = (psize & 0x07ff) * (1 + ((psize >> 11) & 3));
		if (psize > streaming->maxpsize)
			streaming->maxpsize = psize;
	}

	return 0;
}

/* Parse vendor-specific extensions. */
static int uvc_parse_vendor_control(struct uvc_device *dev,
	const unsigned char *buffer, int buflen)
{
	struct usb_device *udev = dev->udev;
	struct uvc_entity *unit;
	unsigned int n, p;
	int handled = 0;

	switch (le16_to_cpu(dev->udev->descriptor.idVendor)) {
	case 0x046d:		/* Logitech */
		if (buffer[1] != 0x41 || buffer[2] != 0x01)
			break;

		/* Logitech implements several vendor specific functions
		 * through vendor specific extension units (LXU).
		 * 
		 * The LXU descriptors are similar to XU descriptors
		 * (see "USB Device Video Class for Video Devices", section
		 * 3.7.2.6 "Extension Unit Descriptor") with the following
		 * differences:
		 * 
		 * ----------------------------------------------------------
		 * 0		bLength		1	 Number
		 *	Size of this descriptor, in bytes: 24+p+n*2
		 * ----------------------------------------------------------
		 * 23+p+n	bmControlsType	N	Bitmap
		 * 	Individual bits in the set are defined:
		 * 	0: Absolute
		 * 	1: Relative
		 *
		 * 	This bitset is mapped exactly the same as bmControls.
		 * ----------------------------------------------------------
		 * 23+p+n*2	bReserved	1	Boolean
		 * ----------------------------------------------------------
		 * 24+p+n*2	iExtension	1	Index
		 *	Index of a string descriptor that describes this
		 *	extension unit.
		 * ----------------------------------------------------------
		 */
		p = buflen >= 22 ? buffer[21] : 0;
		n = buflen >= 25 + p ? buffer[22+p] : 0;

		if (buflen < 25 + p + 2*n) {
			uvc_trace(UVC_TRACE_DESCR, "device %d videocontrol "
			       "interface %d EXTENSION_UNIT error\n", udev->devnum,
			       dev->intf->cur_altsetting->desc.bInterfaceNumber);
			break;
		}

		unit = kzalloc(sizeof *unit + p + 2*n, GFP_KERNEL);
		if (unit == NULL)
			return -ENOMEM;

		unit->id = buffer[3];
		unit->type = VC_EXTENSION_UNIT;
		memcpy(unit->extension.guidExtensionCode, &buffer[4], 16);
		unit->extension.bNumControls = buffer[20];
		unit->extension.bNrInPins = le16_to_cpup((const __le16*)&buffer[21]);
		unit->extension.baSourceID = (__u8*)unit + sizeof *unit;
		memcpy(unit->extension.baSourceID, &buffer[22], p);
		unit->extension.bControlSize = buffer[22+p];
		unit->extension.bmControls = (__u8*)unit + sizeof *unit + p;
		unit->extension.bmControlsType = (__u8*)unit + sizeof *unit + p + n;
		memcpy(unit->extension.bmControls, &buffer[23+p], 2*n);

		if (buffer[24+p+2*n] != 0)
			usb_string(udev, buffer[24+p+2*n], unit->name, sizeof unit->name);
		else
			sprintf(unit->name, "Extension %u", buffer[3]);

		list_add_tail(&unit->list, &dev->entities);
		handled = 1;
		break;
	}

	return handled;
}

static int uvc_parse_control(struct uvc_device *dev)
{
	struct usb_device *udev = dev->udev;
	struct uvc_streaming *streaming;
	struct uvc_entity *unit, *term;
	struct usb_interface *intf;
	struct usb_host_interface *alts = dev->intf->cur_altsetting;
	unsigned char *buffer = alts->extra;
	int buflen = alts->extralen;
	unsigned int i, n, p, len;
	__u16 type;

	/* Parse the default alternate setting only, as the UVC specification
	 * defines a single alternate setting, the default alternate setting
	 * zero.
	 */

	while (buflen > 2) {
		if (uvc_parse_vendor_control(dev, buffer, buflen) ||
		    buffer[1] != USB_DT_CS_INTERFACE)
			goto next_descriptor;

		switch (buffer[2]) {
		case VC_HEADER:
			n = buflen >= 12 ? buffer[11] : 0;

			if (buflen < 12 || buflen < 12 + n) {
				uvc_trace(UVC_TRACE_DESCR, "device %d videocontrol "
				       "interface %d HEADER error\n", udev->devnum,
				       dev->intf->cur_altsetting->desc.bInterfaceNumber);
				return -EINVAL;
			}

			dev->uvc_version = le16_to_cpup((__le16*)&buffer[3]);
			dev->clock_frequency = le32_to_cpup((__le32*)&buffer[7]);

			/* Parse all USB Video Streaming interfaces. */
			for (i = 0; i < n; ++i) {
				intf = usb_ifnum_to_if(udev, buffer[12+i]);
				if (intf == NULL) {
					uvc_trace(UVC_TRACE_DESCR, "device %d "
						"interface %d doesn't exists\n",
						udev->devnum, i);
					continue;
				}

				if (usb_interface_claimed(intf)) {
					uvc_trace(UVC_TRACE_DESCR, "device %d "
						"interface %d is already claimed\n",
						udev->devnum, i);
					continue;
				}

				usb_driver_claim_interface(&uvc_driver.driver, intf, (void*)-1);

				streaming = kzalloc(sizeof *streaming, GFP_KERNEL);
				if (streaming == NULL)
					continue;
				mutex_init(&streaming->mutex);
				streaming->intf = usb_get_intf(intf);
				streaming->intfnum =
					intf->cur_altsetting->desc.bInterfaceNumber;

				if (uvc_parse_streaming(dev, streaming) < 0) {
					usb_put_intf(intf);
					kfree(streaming);
					continue;
				}

				list_add_tail(&streaming->list, &dev->streaming);
			}
			break;

		case VC_INPUT_TERMINAL:
			if (buflen < 8) {
				uvc_trace(UVC_TRACE_DESCR, "device %d videocontrol "
				       "interface %d INPUT_TERMINAL error\n", udev->devnum,
				       dev->intf->cur_altsetting->desc.bInterfaceNumber);
				return -EINVAL;
			}

			/* Make sure the terminal type MSB is not null, otherwise it could be
			 * confused with a unit.
			 */
			type = le16_to_cpup((__le16*)&buffer[4]);
			if ((type & 0xff00) == 0) {
				uvc_trace(UVC_TRACE_DESCR, "device %d videocontrol "
				       "interface %d INPUT_TERMINAL %d has invalid "
				       "type 0x%04x, skipping\n", udev->devnum,
				       dev->intf->cur_altsetting->desc.bInterfaceNumber,
				       buffer[3], type);
				continue;
			}

			n = 0;
			p = 0;
			len = 8;

			if (type == ITT_CAMERA) {
				n = buflen >= 15 ? buffer[14] : 0;
				len = 15;

			} else if (type == ITT_MEDIA_TRANSPORT_INPUT) {
				n = buflen >= 9 ? buffer[8] : 0;
				p = buflen >= 10 + n ? buffer[9+n] : 0;
				len = 10;
			}

			if (buflen < len + n + p) {
				uvc_trace(UVC_TRACE_DESCR, "device %d videocontrol "
				       "interface %d INPUT_TERMINAL error\n", udev->devnum,
				       dev->intf->cur_altsetting->desc.bInterfaceNumber);
				return -EINVAL;
			}

			term = kzalloc(sizeof *term + n + p, GFP_KERNEL);
			if (term == NULL)
				return -ENOMEM;

			term->id = buffer[3];
			term->type = type;

			if (term->type == ITT_CAMERA) {
				term->camera.bControlSize = n;
				term->camera.bmControls = (__u8*)term + sizeof *term;
				term->camera.wObjectiveFocalLengthMin = le16_to_cpup((__le16*)&buffer[8]);
				term->camera.wObjectiveFocalLengthMax = le16_to_cpup((__le16*)&buffer[10]);
				term->camera.wOcularFocalLength = le16_to_cpup((__le16*)&buffer[12]);
				memcpy(term->camera.bmControls, &buffer[15], n);
			} else if (term->type == ITT_MEDIA_TRANSPORT_INPUT) {
				term->media.bControlSize = n;
				term->media.bmControls = (__u8*)term + sizeof *term;
				term->media.bTransportModeSize = p;
				term->media.bmTransportModes = (__u8*)term + sizeof *term + n;
				memcpy(term->media.bmControls, &buffer[9], n);
				memcpy(term->media.bmTransportModes, &buffer[10+n], p);
			}

			if (buffer[7] != 0)
				usb_string(udev, buffer[7], term->name, sizeof term->name);
			else if (term->type == ITT_CAMERA)
				sprintf(term->name, "Camera %u", buffer[3]);
			else if (term->type == ITT_MEDIA_TRANSPORT_INPUT)
				sprintf(term->name, "Media %u", buffer[3]);
			else
				sprintf(term->name, "Input %u", buffer[3]);

			list_add_tail(&term->list, &dev->entities);
			break;

		case VC_OUTPUT_TERMINAL:
			if (buflen < 9) {
				uvc_trace(UVC_TRACE_DESCR, "device %d videocontrol "
				       "interface %d OUTPUT_TERMINAL error\n", udev->devnum,
				       dev->intf->cur_altsetting->desc.bInterfaceNumber);
				return -EINVAL;
			}

			/* Make sure the terminal type MSB is not null, otherwise it could be
			 * confused with a unit.
			 */
			type = le16_to_cpup((__le16*)&buffer[4]);
			if ((type & 0xff00) == 0) {
				uvc_trace(UVC_TRACE_DESCR, "device %d videocontrol "
				       "interface %d OUTPUT_TERMINAL %d has invalid "
				       "type 0x%04x, skipping\n", udev->devnum,
				       dev->intf->cur_altsetting->desc.bInterfaceNumber,
				       buffer[3], type);
				continue;
			}

			term = kzalloc(sizeof *term, GFP_KERNEL);
			if (term == NULL)
				return -ENOMEM;

			term->id = buffer[3];
			term->type = type;
			term->output.bSourceID = buffer[7];

			if (buffer[8] != 0)
				usb_string(udev, buffer[8], term->name, sizeof term->name);
			else
				sprintf(term->name, "Output %u", buffer[3]);

			list_add_tail(&term->list, &dev->entities);
			break;

		case VC_SELECTOR_UNIT:
			p = buflen >= 5 ? buffer[4] : 0;

			if (buflen < 5 || buflen < 6 + p) {
				uvc_trace(UVC_TRACE_DESCR, "device %d videocontrol "
				       "interface %d SELECTOR_UNIT error\n", udev->devnum,
				       dev->intf->cur_altsetting->desc.bInterfaceNumber);
				return -EINVAL;
			}

			unit = kzalloc(sizeof *unit + p, GFP_KERNEL);
			if (unit == NULL)
				return -ENOMEM;

			unit->id = buffer[3];
			unit->type = buffer[2];
			unit->selector.bNrInPins = buffer[4];
			unit->selector.baSourceID = (__u8*)unit + sizeof *unit;
			memcpy(unit->selector.baSourceID, &buffer[5], p);

			if (buffer[5+p] != 0)
				usb_string(udev, buffer[5+p], unit->name, sizeof unit->name);
			else
				sprintf(unit->name, "Selector %u", buffer[3]);

			list_add_tail(&unit->list, &dev->entities);
			break;

		case VC_PROCESSING_UNIT:
			n = buflen >= 8 ? buffer[7] : 0;

			if (buflen < 8 + n) {
				uvc_trace(UVC_TRACE_DESCR, "device %d videocontrol "
				       "interface %d PROCESSING_UNIT error\n", udev->devnum,
				       dev->intf->cur_altsetting->desc.bInterfaceNumber);
				return -EINVAL;
			}

			unit = kzalloc(sizeof *unit + n, GFP_KERNEL);
			if (unit == NULL)
				return -ENOMEM;

			unit->id = buffer[3];
			unit->type = buffer[2];
			unit->processing.bSourceID = buffer[4];
			unit->processing.wMaxMultiplier = le16_to_cpup((__le16*)&buffer[5]);
			unit->processing.bControlSize = buffer[7];
			unit->processing.bmControls = (__u8*)unit + sizeof *unit;
			memcpy(unit->processing.bmControls, &buffer[8], n);
			unit->processing.bmVideoStandards = buffer[9+n];

			if (buffer[8+n] != 0)
				usb_string(udev, buffer[8+n], unit->name, sizeof unit->name);
			else
				sprintf(unit->name, "Processing %u", buffer[3]);

			list_add_tail(&unit->list, &dev->entities);
			break;

		case VC_EXTENSION_UNIT:
			p = buflen >= 22 ? buffer[21] : 0;
			n = buflen >= 24 + p ? buffer[22+p] : 0;

			if (buflen < 24 + p + n) {
				uvc_trace(UVC_TRACE_DESCR, "device %d videocontrol "
				       "interface %d EXTENSION_UNIT error\n", udev->devnum,
				       dev->intf->cur_altsetting->desc.bInterfaceNumber);
				return -EINVAL;
			}

			unit = kzalloc(sizeof *unit + p + n, GFP_KERNEL);
			if (unit == NULL)
				return -ENOMEM;

			unit->id = buffer[3];
			unit->type = buffer[2];
			memcpy(unit->extension.guidExtensionCode, &buffer[4], 16);
			unit->extension.bNumControls = buffer[20];
			unit->extension.bNrInPins = le16_to_cpup((__le16*)&buffer[21]);
			unit->extension.baSourceID = (__u8*)unit + sizeof *unit;
			memcpy(unit->extension.baSourceID, &buffer[22], p);
			unit->extension.bControlSize = buffer[22+p];
			unit->extension.bmControls = (__u8*)unit + sizeof *unit + p;
			memcpy(unit->extension.bmControls, &buffer[23+p], n);

			if (buffer[23+p+n] != 0)
				usb_string(udev, buffer[23+p+n], unit->name, sizeof unit->name);
			else
				sprintf(unit->name, "Extension %u", buffer[3]);

			list_add_tail(&unit->list, &dev->entities);
			break;

		default:
			uvc_trace(UVC_TRACE_DESCR, "Found an unknown CS_INTERFACE "
				"descriptor (%u)\n", buffer[2]);
			break;
		}

next_descriptor:
		buflen -= buffer[0];
		buffer += buffer[0];
	}

	/* Check if the optional status endpoint is present. */
	if (alts->desc.bNumEndpoints == 1) {
		struct usb_host_endpoint *ep = &alts->endpoint[0];
		struct usb_endpoint_descriptor *desc = &ep->desc;

		if (usb_endpoint_is_int_in(desc) &&
		    le16_to_cpu(desc->wMaxPacketSize) >= 8 &&
		    desc->bInterval != 0) {
			uvc_trace(UVC_TRACE_DESCR, "Found a Status endpoint "
				"(addr %02x).\n", desc->bEndpointAddress);
			dev->int_ep = ep;
		}
	}

	return 0;
}

/* ------------------------------------------------------------------------
 * USB probe and disconnect
 */

/*
 * Unregister the video devices.
 */
static void uvc_unregister_video(struct uvc_device *dev)
{
	if (dev->video.vdev) {
		if (dev->video.vdev->minor == -1)
			video_device_release(dev->video.vdev);
		else
			video_unregister_device(dev->video.vdev);
		dev->video.vdev = NULL;
	}
}

/*
 * Scan the UVC descriptors to locate a chain starting at an Output Terminal
 * and containing the following units:
 *
 * - a USB Streaming Output Terminal
 * - zero or one Processing Unit
 * - zero, one or mode single-input Selector Units
 * - zero or one multiple-input Selector Units, provided all inputs are
 *   connected to input terminals
 * - zero, one or mode single-input Extension Units
 * - one Camera Input Terminal, or one or more External terminals.
 *
 * A side forward scan is made on each detected entity to check for additional
 * extension units.
 */
static int uvc_scan_chain_entity(struct uvc_video_device *video,
	struct uvc_entity *entity)
{
	switch (entity->type) {
	case VC_EXTENSION_UNIT:
		if (uvc_trace_param & UVC_TRACE_PROBE)
			printk(" <- XU %d", entity->id);

		if (entity->extension.bNrInPins != 1) {
			uvc_trace(UVC_TRACE_DESCR, "Extension unit %d has more "
				"than 1 input pin.\n", entity->id);
			return -1;
		}

		list_add_tail(&entity->chain, &video->extensions);
		break;

	case VC_PROCESSING_UNIT:
		if (uvc_trace_param & UVC_TRACE_PROBE)
			printk(" <- PU %d", entity->id);

		if (video->processing != NULL) {
			uvc_trace(UVC_TRACE_DESCR, "Found multiple "
				"Processing Units in chain.\n");
			return -1;
		}

		video->processing = entity;
		break;

	case VC_SELECTOR_UNIT:
		if (uvc_trace_param & UVC_TRACE_PROBE)
			printk(" <- SU %d", entity->id);

		/* Single-input selector units are ignored. */
		if (entity->selector.bNrInPins == 1)
			break;

		if (video->selector != NULL) {
			uvc_trace(UVC_TRACE_DESCR, "Found multiple Selector "
				"Units in chain.\n");
			return -1;
		}

		video->selector = entity;
		break;

	case ITT_VENDOR_SPECIFIC:
	case ITT_CAMERA:
	case ITT_MEDIA_TRANSPORT_INPUT:
		if (uvc_trace_param & UVC_TRACE_PROBE)
			printk(" <- IT %d\n", entity->id);

		list_add_tail(&entity->chain, &video->iterms);
		break;

	default:
		uvc_trace(UVC_TRACE_DESCR, "Unsupported entity type "
			"0x%04x found in chain.n", entity->type);
		return -1;
	}

	return 0;
}

static int uvc_scan_chain_forward(struct uvc_video_device *video,
	struct uvc_entity *entity, struct uvc_entity *prev)
{
	struct uvc_entity *forward;
	int found;

	/* Forward scan */
	forward = NULL;
	found = 0;

	while (1) {
		forward = uvc_entity_by_reference(video->dev, entity->id,
			forward);
		if (forward == NULL)
			break;

		if (forward == prev || forward->type != VC_EXTENSION_UNIT)
			continue;

		if (forward->extension.bNrInPins != 1) {
			uvc_trace(UVC_TRACE_DESCR, "Extension unit %d has"
				"more than 1 input pin.\n", entity->id);
			return -1;
		}

		list_add_tail(&forward->chain, &video->extensions);
		if (uvc_trace_param & UVC_TRACE_PROBE) {
			if (!found)
				printk(" (-> XU");

			printk(" %d", forward->id);
			found = 1;
		}
	}
	if (found)
		printk(")");

	return 0;
}

static int uvc_scan_chain_backward(struct uvc_video_device *video,
	struct uvc_entity *entity)
{
	struct uvc_entity *term;
	int id = -1, i;

	switch (entity->type) {
	case VC_EXTENSION_UNIT:
		id = entity->extension.baSourceID[0];
		break;

	case VC_PROCESSING_UNIT:
		id = entity->processing.bSourceID;
		break;

	case VC_SELECTOR_UNIT:
		/* Single-input selector units are ignored. */
		if (entity->selector.bNrInPins == 1) {
			id = entity->selector.baSourceID[0];
			break;
		}

		if (uvc_trace_param & UVC_TRACE_PROBE)
			printk(" <- IT");

		video->selector = entity;
		for (i = 0; i < entity->selector.bNrInPins; ++i) {
			id = entity->selector.baSourceID[i];
			term = uvc_entity_by_id(video->dev, id);
			if (term == NULL || !UVC_ENTITY_IS_ITERM(term)) {
				uvc_trace(UVC_TRACE_DESCR, "Selector unit %d "
					"input %d isn't connected to an "
					"input terminal\n", entity->id, i);
				return -1;
			}

			if (uvc_trace_param & UVC_TRACE_PROBE)
				printk(" %d", term->id);

			list_add_tail(&term->chain, &video->iterms);
			uvc_scan_chain_forward(video, term, entity);
		}

		if (uvc_trace_param & UVC_TRACE_PROBE)
			printk("\n");

		id = 0;
		break;
	}

	return id;
}

static int uvc_scan_chain(struct uvc_video_device *video)
{
	struct uvc_entity *entity, *prev;
	int id;

	entity = video->oterm;
	uvc_trace(UVC_TRACE_PROBE, "Scanning UVC chain: OT %d", entity->id);
	id = entity->output.bSourceID;
	while (id != 0) {
		prev = entity;
		entity = uvc_entity_by_id(video->dev, id);
		if (entity == NULL) {
			uvc_trace(UVC_TRACE_DESCR, "Found reference to "
				"unknown entity %d.\n", id);
			return -1;
		}

		/* Process entity */
		if (uvc_scan_chain_entity(video, entity) < 0)
			return -1;

		/* Forward scan */
		if (uvc_scan_chain_forward(video, entity, prev) < 0)
			return -1;

		/* Stop when a terminal is found. */
		if (!UVC_ENTITY_IS_UNIT(entity))
			break;

		/* Backward scan */
		id = uvc_scan_chain_backward(video, entity);
		if (id < 0)
			return id;
	}

	/* Initialize the video buffers queue. */
	uvc_queue_init(&video->queue);

	return 0;
}

/*
 * Register the video devices.
 *
 * The driver currently supports a single video device per control interface
 * only. The terminal and units must match the following structure:
 *
 * ITT_CAMERA -> VC_PROCESSING_UNIT -> VC_EXTENSION_UNIT{0,n} -> TT_STREAMING
 *
 * The Extension Units, if present, must have a single input pin. The
 * Processing Unit and Extension Units can be in any order. Additional
 * Extension Units connected to the main chain as single-unit branches are
 * also supported.
 */
static int uvc_register_video(struct uvc_device *dev)
{
	struct video_device *vdev;
	struct uvc_entity *term;
	int found = 0, ret;

	/* Check if the control interface matches the structure we expect. */
	list_for_each_entry(term, &dev->entities, list) {
		struct list_head *ps;

		if (term->type != TT_STREAMING)
			continue;

		memset(&dev->video, 0, sizeof dev->video);
		mutex_init(&dev->video.ctrl_mutex);
		INIT_LIST_HEAD(&dev->video.iterms);
		INIT_LIST_HEAD(&dev->video.extensions);
		dev->video.oterm = term;
		dev->video.dev = dev;
		if (uvc_scan_chain(&dev->video) < 0)
			continue;

		list_for_each(ps, &dev->streaming) {
			struct uvc_streaming *streaming;

			streaming = list_entry(ps, struct uvc_streaming, list);
			if (streaming->input.bTerminalLink == term->id) {
				dev->video.streaming = streaming;
				found = 1;
				break;
			}
		}

		if (found)
			break;
	}

	if (!found) {
		uvc_printk(KERN_INFO, "No valid video chain found.\n");
		return -1;
	}

	if (uvc_trace_param & UVC_TRACE_PROBE) {
		uvc_printk(KERN_INFO, "Found a valid video chain (");
		list_for_each_entry(term, &dev->video.iterms, chain) {
			printk("%d", term->id);
			if (term->chain.next != &dev->video.iterms)
				printk(",");
		}
		printk(" -> %d).\n", dev->video.oterm->id);
	}

	/* Initialize the streaming interface with default streaming
	 * parameters.
	 */
	if ((ret = uvc_video_init(&dev->video)) < 0) {
		uvc_printk(KERN_ERR, "Failed to initialize the device "
			"(%d).\n", ret);
		return ret;
	}

	/* Register the device with V4L. */
	vdev = video_device_alloc();
	if (vdev == NULL)
		return -1;

	sprintf(vdev->name, "USB Video Class");
	/* We already hold a reference to dev->udev. The video device will be
	 * unregistered before the reference is released, so we don't need to
	 * get another one.
	 */
	vdev->dev = &dev->udev->dev;
	vdev->type = 0;
	vdev->type2 = 0;
	vdev->hardware = 0;
	vdev->minor = -1;
	vdev->fops = &uvc_fops;
	vdev->release = video_device_release;

	/* Set the driver data before calling video_register_device, otherwise
	 * uvc_v4l2_open might race us.
	 *
	 * FIXME: usb_set_intfdata hasn't been called so far. Is that a
	 * 	  problem ? Does any function which could be called here get
	 * 	  a pointer to the usb_interface ?
	 */
	dev->video.vdev = vdev;
	video_set_drvdata(vdev, &dev->video);

	if (video_register_device(vdev, VFL_TYPE_GRABBER, -1) < 0) {
		dev->video.vdev = NULL;
		video_device_release(vdev);
		return -1;
	}

	return 0;
}

/*
 * Delete the UVC device.
 *
 * Called by the kernel when the last reference to the uvc_device structure
 * is released.
 *
 * Unregistering the video devices is done here because every opened instance
 * must be closed before the device can be unregistered. An alternative would
 * have been to use another reference count for uvc_v4l2_open/uvc_release, and
 * unregister the video devices on disconnect when that reference count drops
 * to zero.
 *
 * As this function is called after or during disconnect(), all URBs have
 * already been canceled by the USB core. There is no need to kill the
 * interrupt URB manually.
 */
void uvc_delete(struct kref *kref)
{
	struct uvc_device *dev = container_of(kref, struct uvc_device, kref);
	struct list_head *p, *n;

	/* Unregister the video device */
	uvc_unregister_video(dev);
	usb_put_intf(dev->intf);
	usb_put_dev(dev->udev);

	uvc_ctrl_cleanup_device(dev);

	list_for_each_safe(p, n, &dev->entities) {
		struct uvc_entity *entity;
		entity = list_entry(p, struct uvc_entity, list);
		kfree(entity);
	}

	list_for_each_safe(p, n, &dev->streaming) {
		struct uvc_streaming *streaming;
		streaming = list_entry(p, struct uvc_streaming, list);
		usb_put_intf(streaming->intf);
		kfree(streaming->format);
		kfree(streaming->input.bmaControls);
		kfree(streaming);
	}

	kfree(dev);
}

static int uvc_probe(struct usb_interface *intf,
		     const struct usb_device_id *id)
{
	struct usb_device *udev = interface_to_usbdev(intf);
	struct uvc_device *dev;
	int ret;

	if (id->idVendor && id->idProduct)
		uvc_trace(UVC_TRACE_PROBE, "Probing known UVC device %s "
				"(%04x:%04x)\n", udev->devpath, id->idVendor,
				id->idProduct);
	else
		uvc_trace(UVC_TRACE_PROBE, "Probing generic UVC device %s\n",
				udev->devpath);

	/* Allocate memory for the device and initialize it */
	if ((dev = kzalloc(sizeof *dev, GFP_KERNEL)) == NULL)
		return -ENOMEM;

	INIT_LIST_HEAD(&dev->entities);
	INIT_LIST_HEAD(&dev->streaming);
	kref_init(&dev->kref);

	dev->udev = usb_get_dev(udev);
	dev->intf = usb_get_intf(intf);
	dev->intfnum = intf->cur_altsetting->desc.bInterfaceNumber;
	dev->quirks = id->driver_info;

	/* Parse the Video Class control descriptor */
	if (uvc_parse_control(dev) < 0) {
		uvc_trace(UVC_TRACE_PROBE, "Unable to parse UVC descriptors.\n");
		goto error;
	}

	uvc_printk(KERN_INFO, "Found UVC %u.%02u device %s (%04x:%04x)\n",
		dev->uvc_version >> 8, dev->uvc_version & 0xff,
		udev->product ? udev->product : "<unnamed>",
		le16_to_cpu(udev->descriptor.idVendor),
		le16_to_cpu(udev->descriptor.idProduct));

	/* Initialize controls */
	if (uvc_ctrl_init_device(dev) < 0)
		goto error;

	/* Register the video devices */
	if (uvc_register_video(dev) < 0)
		goto error;

	/* Save our data pointer in the interface data */
	usb_set_intfdata(intf, dev);

	/* Initialize the interrupt URB */
	if (dev->int_ep != NULL && (ret = uvc_init_status(dev)) < 0) {
		uvc_printk(KERN_INFO, "Unable to initialize the status "
			"endpoint (%d), status interrupt will not be "
			"supported.\n", ret);
	}

	uvc_trace(UVC_TRACE_PROBE, "UVC device initialized.\n");
	return 0;

error:
	kref_put(&dev->kref, uvc_delete);
	return -ENODEV;
}

static void uvc_disconnect(struct usb_interface *intf)
{
	struct uvc_device *dev = usb_get_intfdata(intf);

	if (dev == (void*)-1)
		return;

	/* Set the USB interface data to NULL. This can be done outside the
	 * lock, as there's no other reader.
	 */
	usb_set_intfdata(intf, NULL);

	/* uvc_v4l2_open() might race uvc_disconnect(). A static driver-wide
	 * lock is needed to prevent uvc_disconnect from releasing its
	 * reference to the uvc_device instance after uvc_v4l2_open() received
	 * the pointer to the device (video_devdata) but before it got the
	 * chance to increase the reference count (kref_get). An alternative
	 * (used by the usb-skeleton driver) would have been to use the big
	 * kernel lock instead of a driver-specific mutex (open calls on
	 * char devices are protected by the BKL), but that approach is not
	 * recommended for new code.
	 */
	mutex_lock(&uvc_driver.open_mutex);

	dev->state |= UVC_DEV_DISCONNECTED;
	kref_put(&dev->kref, uvc_delete);

	mutex_unlock(&uvc_driver.open_mutex);
}

/* ------------------------------------------------------------------------
 * Driver initialization and cleanup
 */

/*
 * The Logitech cameras listed below have their interface class set to
 * VENDOR_SPEC because they don't announce themselves as UVC devices, even
 * though they are compliant.
 */
static struct usb_device_id uvc_ids[] = {
	/* Microsoft Lifecam NX-6000 */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x045e,
	  .idProduct		= 0x00f8,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_QUIRK_PROBE_MINMAX },
	/* Logitech Quickcam Fusion */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x046d,
	  .idProduct		= 0x08c1,
	  .bInterfaceClass	= USB_CLASS_VENDOR_SPEC,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0 },
	/* Logitech Quickcam Orbit MP */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x046d,
	  .idProduct		= 0x08c2,
	  .bInterfaceClass	= USB_CLASS_VENDOR_SPEC,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0 },
	/* Logitech Quickcam Pro for Notebook */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x046d,
	  .idProduct		= 0x08c3,
	  .bInterfaceClass	= USB_CLASS_VENDOR_SPEC,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0 },
	/* Logitech Quickcam Pro 5000 */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x046d,
	  .idProduct		= 0x08c5,
	  .bInterfaceClass	= USB_CLASS_VENDOR_SPEC,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0 },
	/* Logitech Quickcam OEM Dell Notebook */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x046d,
	  .idProduct		= 0x08c6,
	  .bInterfaceClass	= USB_CLASS_VENDOR_SPEC,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0 },
	/* Logitech Quickcam OEM Cisco VT Camera II */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x046d,
	  .idProduct		= 0x08c7,
	  .bInterfaceClass	= USB_CLASS_VENDOR_SPEC,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0 },
	/* Bodelin ProScopeHR */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x19ab,
	  .idProduct		= 0x1000,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_QUIRK_STATUS_INTERVAL
	},
	/* Creative Live! Optia */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x041e,
	  .idProduct		= 0x4057,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_QUIRK_PROBE_MINMAX
	},
	/* Ecamm Pico iMage */
	{ .match_flags		= USB_DEVICE_ID_MATCH_DEVICE
				| USB_DEVICE_ID_MATCH_INT_INFO,
	  .idVendor		= 0x18cd,
	  .idProduct		= 0xcafe,
	  .bInterfaceClass	= USB_CLASS_VIDEO,
	  .bInterfaceSubClass	= 1,
	  .bInterfaceProtocol	= 0,
	  .driver_info		= UVC_QUIRK_PROBE_EXTRAFIELDS
	},
	/* Generic USB Video Class */
	{ USB_INTERFACE_INFO(USB_CLASS_VIDEO, 1, 0) },
	{}
};

MODULE_DEVICE_TABLE(usb, uvc_ids);

struct uvc_driver uvc_driver = {
	.driver = {
#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,16)
		.owner		= THIS_MODULE,
#endif
		.name		= "uvcvideo",
		.probe		= uvc_probe,
		.disconnect	= uvc_disconnect,
		.id_table	= uvc_ids,
	},
};

static int __init uvc_init(void)
{
	int result;

	INIT_LIST_HEAD(&uvc_driver.devices);
	INIT_LIST_HEAD(&uvc_driver.controls);
	mutex_init(&uvc_driver.open_mutex);
	mutex_init(&uvc_driver.ctrl_mutex);

	uvc_ctrl_init();

	result = usb_register(&uvc_driver.driver);
	if (result == 0)
		printk(KERN_INFO DRIVER_DESC " (v" DRIVER_VERSION ")\n");
	return result;
}

static void __exit uvc_cleanup(void)
{
	usb_deregister(&uvc_driver.driver);
}

module_init(uvc_init);
module_exit(uvc_cleanup);

unsigned int uvc_trace_param = 0;
module_param_named(trace, uvc_trace_param, uint, S_IRUGO|S_IWUSR);

MODULE_AUTHOR(DRIVER_AUTHOR);
MODULE_DESCRIPTION(DRIVER_DESC);
MODULE_LICENSE("GPL");

