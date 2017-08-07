from pico_status cimport PICO_STATUS, PICO_INFO

cdef extern from 'usbPT104Api.h':
    cdef int USBPT104_MIN_WIRES = 2
    cdef int USBPT104_MAX_WIRES = 4

    ctypedef enum USBPT104_CHANNELS:
        USBPT104_CHANNEL_1 = 1,
        USBPT104_CHANNEL_2,
        USBPT104_CHANNEL_3,
        USBPT104_CHANNEL_4,
        USBPT104_CHANNEL_5,
        USBPT104_CHANNEL_6,
        USBPT104_CHANNEL_7,
        USBPT104_CHANNEL_8,
        USBPT104_MAX_CHANNELS = USBPT104_CHANNEL_8


    ctypedef enum USBPT104_DATA_TYPES:
        USBPT104_OFF,
        USBPT104_PT100,
        USBPT104_PT1000,
        USBPT104_RESISTANCE_TO_375R,
        USBPT104_RESISTANCE_TO_10K,
        USBPT104_DIFFERENTIAL_TO_115MV,
        USBPT104_DIFFERENTIAL_TO_2500MV,
        USBPT104_SINGLE_ENDED_TO_115MV,
        USBPT104_SINGLE_ENDED_TO_2500MV,
        USBPT104_MAX_DATA_TYPES

    ctypedef enum IP_DETAILS_TYPE:
        IDT_GET,
        IDT_SET

    ctypedef enum COMMUNICATION_TYPE:
        CT_USB = 0x00000001,
        CT_ETHERNET = 0x00000002,
        CT_ALL = 0xFFFFFFFF

        
    PICO_STATUS UsbPt104Enumerate(
        char*   details,
        unsigned int* length,
        COMMUNICATION_TYPE type) nogil

    PICO_STATUS UsbPt104OpenUnit(
        short* handle,
        char* serial) nogil

    PICO_STATUS UsbPt104OpenUnitViaIp(
        short* handle,
        char*  serial,
        char*  ipAddress) nogil

    PICO_STATUS UsbPt104CloseUnit(
        short handle) nogil

    PICO_STATUS UsbPt104IpDetails(
        short   handle,
        short*  enabled,
        char*   ipaddress,
        unsigned short* length,
        unsigned short* listeningPort,
        IP_DETAILS_TYPE type) nogil

    PICO_STATUS UsbPt104GetUnitInfo(
        short    handle,
        char*    string,
        short    stringLength,
        short*   requiredSize,
        PICO_INFO  info) nogil

    PICO_STATUS UsbPt104SetChannel(
        short              handle,
        USBPT104_CHANNELS    channel,
        USBPT104_DATA_TYPES  type,
        short              noOfWires) nogil

    PICO_STATUS UsbPt104SetMains(
        short  handle,
        unsigned short sixtyHertz) nogil

    PICO_STATUS UsbPt104GetValue(
        short            handle,
        USBPT104_CHANNELS  channel,
        int*             value,
        short            filtered) nogil