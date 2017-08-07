from .pico_status cimport check_status
from .pico_status import PicoError
from . cimport pico_status
from .frozendict import frozendict
from cpython.mem cimport PyMem_Malloc, PyMem_Free

channel_dict = frozendict({
    '1' : USBPT104_CHANNEL_1,
    '2' : USBPT104_CHANNEL_2,
    '3' : USBPT104_CHANNEL_3,
    '4' : USBPT104_CHANNEL_4,
    '5' : USBPT104_CHANNEL_5,
    '6' : USBPT104_CHANNEL_6,
    '7' : USBPT104_CHANNEL_7,
    '8' : USBPT104_CHANNEL_8,
    'max' : USBPT104_MAX_CHANNELS})

datatype_dict = frozendict({
    'off' : USBPT104_OFF,
    'PT100' : USBPT104_PT100,
    'PT1000' : USBPT104_PT1000,
    'RES375' : USBPT104_RESISTANCE_TO_375R,
    'RES10K' : USBPT104_RESISTANCE_TO_10K,
    'DIF115' : USBPT104_DIFFERENTIAL_TO_115MV,
    'DIF2500' : USBPT104_DIFFERENTIAL_TO_2500MV,
    'SE115' : USBPT104_SINGLE_ENDED_TO_115MV,
    'SE2500' : USBPT104_SINGLE_ENDED_TO_2500MV,
    'max' : USBPT104_MAX_DATA_TYPES})

ipdetail_dict = frozendict({
    'get' : IDT_GET,
    'set' : IDT_SET})

comtype_dict = frozendict({
    'usb' : CT_USB,
    'eth' : CT_ETHERNET,
    'all' : CT_ALL})
    
    
cdef enumerate(char* details, unsigned int length, comm_type):
    cdef COMMUNICATION_TYPE comm = comtype_dict[comm_type]
    with nogil:
        status = UsbPt104Enumerate(details, &length, comm)
    check_status(status)
    
cdef open_unit(char *serial_str):
    cdef PICO_STATUS status
    cdef short handle

    with nogil:
        status = UsbPt104OpenUnit(&handle, serial_str)
    check_status(status)

    return handle

cdef open_unit_ip(char *serial_str, char* ipAddress):
    cdef PICO_STATUS status
    cdef short handle

    with nogil:
        status = UsbPt104OpenUnitViaIp(&handle, serial_str, ipAddress)
    check_status(status)

    return handle
    
cdef close_unit(short handle):
    with nogil:
        status = UsbPt104CloseUnit(handle)
    check_status(status)

cdef get_unit_info(short handle, PICO_INFO info):

    cdef short n = 20
    cdef char* info_str = <char *>PyMem_Malloc(sizeof(char)*(n))
    cdef short required_n

    cdef bytes py_info_str
    
    cdef PICO_STATUS status

    with nogil:
        status = UsbPt104GetUnitInfo(handle, info_str, n, &required_n, info)
    check_status(status)
    
    # Make sure we had a big enough string
    if required_n > n:
        PyMem_Free(info_str)
        n = required_n
        info_str = <char *>PyMem_Malloc(sizeof(char)*(n))
        
        with nogil:
            status = UsbPt104GetUnitInfo(handle, info_str, n, &required_n, info)
        check_status(status)

    try:
        py_info_str = info_str
    finally:
        PyMem_Free(info_str)

    return py_info_str
    
    
cdef set_channel(short handle, channel, type, short noOfWires):
    cdef USBPT104_CHANNELS ch = channel_dict[channel]
    cdef USBPT104_DATA_TYPES ty = datatype_dict[type]
    cdef PICO_STATUS status
    with nogil:
        status = UsbPt104SetChannel(handle, ch, ty, noOfWires)
    check_status(status)
    
cdef set_mains(short handle, unsigned short sixtyHertz):
    cdef PICO_STATUS status
    with nogil:
        status = UsbPt104SetMains(handle, sixtyHertz)
    check_status(status)
    
cdef get_value(short handle, channel, short filtered):
    cdef USBPT104_CHANNELS ch = channel_dict[channel]
    cdef PICO_STATUS status
    cdef int value
    with nogil:
        status = UsbPt104GetValue(handle, ch, &value, filtered)
    check_status(status)
    
    return value
   
cdef class Pt104:
    cdef short _handle
    cdef object __channel_states
    cdef object __serial_string
       
    def __cinit__(self, serial = None, channel_configs = None, mains = 0):
    
        cdef char* serial_str
        if serial == None:
            serial_str = NULL
        else:
            serial_str = serial
        
        self._handle = open_unit(serial_str)
    
    def __init__(self, serial = None, channel_configs = None):
        self.__serial_string = get_unit_info(
                self._handle, pico_status.PICO_BATCH_AND_SERIAL)
                
        default_configs = {'1': {'Data_Type': 'PT100', 'No Wires': 4, "Filter": 1}}
        self.__channel_states = {}
        if channel_configs is None:
            channel_configs = default_configs
        for channel, config in channel_configs.items():
            set_channel(self._handle, channel, config['Data_Type'], config['No Wires'])
            self.__channel_states[channel] = config
    
    def get_value(self, channel):
        raw = get_value(self._handle, channel, self.__channel_states[channel]['Filter'])
        data_type = self.__channel_states[channel]['Data_Type']
        if data_type in ['PT100', 'PT1000']:
            converted = raw/1000
        elif data_type in ['DIF2500', 'SE2500']:
            converted = raw/(1e-8)
        elif data_type in ['DIF115', 'SE115']:
            converted = raw/(1e-6)
        elif data_type == 'RES375':
            converted = raw/(1e-3)
        return converted

    def __dealloc__(self):
        try:
            close_unit(self._handle)
        except KeyError:
            pass




        
        
        
    
    
