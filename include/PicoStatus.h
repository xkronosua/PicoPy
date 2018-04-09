/**************************************************************************
*
* Filename:    pico_status.h
* Copyright:   Pico Technology Limited 2002 - 2016
* Author:      MAS
* Description:
*
* This header defines the status codes returned by a
*	Pico device, a PC Oscilloscope or data logger.
*
* In comments, "<API>" is a placeholder for the name of the scope or
* data logger API. For example, for the ps5000a API, it stands for
* "PS5000A" or "ps5000a".
*
*************************************************************************/
#ifndef __PICOSTATUS_H__
#define __PICOSTATUS_H__

#include <stdint.h>

#define	PICO_DRIVER_VERSION                         0x00000000UL
#define	PICO_USB_VERSION                            0x00000001UL
#define	PICO_HARDWARE_VERSION                       0x00000002UL
#define	PICO_VARIANT_INFO                           0x00000003UL
#define	PICO_BATCH_AND_SERIAL                       0x00000004UL
#define	PICO_CAL_DATE                               0x00000005UL
#define	PICO_KERNEL_VERSION                         0x00000006UL

#define PICO_DIGITAL_HARDWARE_VERSION               0x00000007UL
#define PICO_ANALOGUE_HARDWARE_VERSION              0x00000008UL

#define PICO_FIRMWARE_VERSION_1                     0x00000009UL
#define PICO_FIRMWARE_VERSION_2                     0x0000000AUL

#define PICO_MAC_ADDRESS                            0x0000000BUL

#define PICO_SHADOW_CAL                             0x0000000CUL

#define PICO_IPP_VERSION                            0x0000000DUL

#define PICO_DRIVER_PATH                            0x0000000EUL

typedef uint32_t PICO_INFO;

// The PicoScope is functioning correctly.
#define PICO_OK                                     0x00000000UL

// An attempt has been made to open more than <API>_MAX_UNITS.
#define PICO_MAX_UNITS_OPENED                       0x00000001UL

// Not enough memory could be allocated on the host machine.
#define PICO_MEMORY_FAIL                            0x00000002UL

// No Pico Technology device could be found.
#define PICO_NOT_FOUND                              0x00000003UL

// Unable to download firmware.
#define PICO_FW_FAIL                                0x00000004UL

// The driver is busy opening a device.
#define PICO_OPEN_OPERATION_IN_PROGRESS             0x00000005UL

// An unspecified failure occurred.
#define PICO_OPERATION_FAILED                       0x00000006UL

// The PicoScope is not responding to commands from the PC.
#define PICO_NOT_RESPONDING                         0x00000007UL

// The configuration information in the PicoScope is corrupt or missing.
#define PICO_CONFIG_FAIL                            0x00000008UL

// The picopp.sys file is too old to be used with the device driver.
#define PICO_KERNEL_DRIVER_TOO_OLD                  0x00000009UL

// The EEPROM has become corrupt, so the device will use a default setting.
#define PICO_EEPROM_CORRUPT                         0x0000000AUL

// The operating system on the PC is not supported by this driver.
#define PICO_OS_NOT_SUPPORTED                       0x0000000BUL

// There is no device with the handle value passed.
#define PICO_INVALID_HANDLE                         0x0000000CUL

// A parameter value is not valid.
#define PICO_INVALID_PARAMETER                      0x0000000DUL

// The timebase is not supported or is invalid.
#define PICO_INVALID_TIMEBASE                       0x0000000EUL

// The voltage range is not supported or is invalid.
#define PICO_INVALID_VOLTAGE_RANGE                  0x0000000FUL

// The channel number is not valid on this device or no channels have been set.
#define PICO_INVALID_CHANNEL                        0x00000010UL

// The channel set for a trigger is not available on this device.
#define PICO_INVALID_TRIGGER_CHANNEL                0x00000011UL

// The channel set for a condition is not available on this device.
#define PICO_INVALID_CONDITION_CHANNEL              0x00000012UL

// The device does not have a signal generator.
#define PICO_NO_SIGNAL_GENERATOR                    0x00000013UL

// Streaming has failed to start or has stopped without user request.
#define PICO_STREAMING_FAILED                       0x00000014UL

// Block failed to start - a parameter may have been set wrongly.
#define PICO_BLOCK_MODE_FAILED                      0x00000015UL

// A parameter that was required is NULL.
#define PICO_NULL_PARAMETER                         0x00000016UL

// The current functionality is not available while using ETS capture mode.
#define PICO_ETS_MODE_SET                           0x00000017UL

// No data is available from a run block call.
#define PICO_DATA_NOT_AVAILABLE                     0x00000018UL

// The buffer passed for the information was too small.
#define PICO_STRING_BUFFER_TO_SMALL                 0x00000019UL

// ETS is not supported on this device.
#define PICO_ETS_NOT_SUPPORTED                      0x0000001AUL

// The auto trigger time is less than the time it will take to collect the pre-trigger data.
#define PICO_AUTO_TRIGGER_TIME_TO_SHORT             0x0000001BUL

// The collection of data has stalled as unread data would be overwritten.
#define PICO_BUFFER_STALL                           0x0000001CUL

// Number of samples requested is more than available in the current memory segment.
#define PICO_TOO_MANY_SAMPLES                       0x0000001DUL

// Not possible to create number of segments requested.
#define PICO_TOO_MANY_SEGMENTS                      0x0000001EUL

// A null pointer has been passed in the trigger function or one of the parameters is out of range.
#define PICO_PULSE_WIDTH_QUALIFIER                  0x0000001FUL

// One or more of the hold-off parameters are out of range.
#define PICO_DELAY                                  0x00000020UL

// One or more of the source details are incorrect.
#define	PICO_SOURCE_DETAILS                         0x00000021UL

// One or more of the conditions are incorrect.
#define PICO_CONDITIONS                             0x00000022UL

// The driver's thread is currently in the <API>Ready callback 
// function and therefore the action cannot be carried out.
#define	PICO_USER_CALLBACK                          0x00000023UL

// An attempt is being made to get stored data while streaming. Either stop
// streaming by calling <API>Stop, or use <API>GetStreamingLatestValues.
#define PICO_DEVICE_SAMPLING                        0x00000024UL

// Data is unavailable because a run has not been completed.
#define PICO_NO_SAMPLES_AVAILABLE                   0x00000025UL

// The memory segment index is out of range.
#define PICO_SEGMENT_OUT_OF_RANGE                   0x00000026UL

// The device is busy so data cannot be returned yet.
#define PICO_BUSY                                   0x00000027UL

// The start time to get stored data is out of range.
#define PICO_STARTINDEX_INVALID                     0x00000028UL

// The information number requested is not a valid number.
#define PICO_INVALID_INFO                           0x00000029UL

// The handle is invalid so no information is available about the device. 
// Only PICO_DRIVER_VERSION is available.
#define PICO_INFO_UNAVAILABLE                       0x0000002AUL

// The sample interval selected for streaming is out of range.
#define PICO_INVALID_SAMPLE_INTERVAL                0x0000002BUL

// ETS is set but no trigger has been set. A trigger setting is required for ETS.
#define PICO_TRIGGER_ERROR                          0x0000002CUL

// Driver cannot allocate memory.
#define PICO_MEMORY                                 0x0000002DUL

// Incorrect parameter passed to the signal generator.
#define PICO_SIG_GEN_PARAM                          0x0000002EUL

// Conflict between the shots and sweeps parameters sent to the signal generator.
#define PICO_SHOTS_SWEEPS_WARNING                   0x0000002FUL

// A software trigger has been sent but the trigger source is not a software trigger.
#define PICO_SIGGEN_TRIGGER_SOURCE                  0x00000030UL

// An <API>SetTrigger call has found a conflict between the trigger source and the AUX output enable.
#define PICO_AUX_OUTPUT_CONFLICT                    0x00000031UL

// ETS mode is being used and AUX is set as an input.
#define PICO_AUX_OUTPUT_ETS_CONFLICT                0x00000032UL

// Attempt to set different EXT input thresholds set for signal generator and oscilloscope trigger.
#define PICO_WARNING_EXT_THRESHOLD_CONFLICT         0x00000033UL

// An <API>SetTrigger... function has set AUX as an output and the signal generator is using it as a trigger.
#define PICO_WARNING_AUX_OUTPUT_CONFLICT            0x00000034UL

// The combined peak to peak voltage and the analog offset voltage exceed the maximum voltage the signal generator can produce.
#define PICO_SIGGEN_OUTPUT_OVER_VOLTAGE             0x00000035UL

// NULL pointer passed as delay parameter.
#define PICO_DELAY_NULL                             0x00000036UL

// The buffers for overview data have not been set while streaming.
#define PICO_INVALID_BUFFER                         0x00000037UL

// The analog offset voltage is out of range.
#define PICO_SIGGEN_OFFSET_VOLTAGE                  0x00000038UL

// The analog peak-to-peak voltage is out of range.
#define PICO_SIGGEN_PK_TO_PK                        0x00000039UL

// A block collection has been cancelled.
#define PICO_CANCELLED                              0x0000003AUL

// The segment index is not currently being used.
#define PICO_SEGMENT_NOT_USED                       0x0000003BUL

// The wrong GetValues function has been called for the collection mode in use.
#define PICO_INVALID_CALL                           0x0000003CUL

#define PICO_GET_VALUES_INTERRUPTED                 0x0000003DUL

// The function is not available.
#define PICO_NOT_USED                               0x0000003FUL

// The aggregation ratio requested is out of range.
#define PICO_INVALID_SAMPLERATIO                    0x00000040UL

// Device is in an invalid state.
#define PICO_INVALID_STATE                          0x00000041UL

// The number of segments allocated is fewer than the number of captures requested.
#define PICO_NOT_ENOUGH_SEGMENTS                    0x00000042UL

// A driver function has already been called and not yet finished.
// Only one call to the driver can be made at any one time.
#define PICO_DRIVER_FUNCTION                        0x00000043UL

// Not used
#define PICO_RESERVED                               0x00000044UL

// An invalid coupling type was specified in <API>SetChannel.
#define PICO_INVALID_COUPLING                       0x00000045UL

// An attempt was made to get data before a data buffer was defined.
#define PICO_BUFFERS_NOT_SET                        0x00000046UL

// The selected downsampling mode (used for data reduction) is not allowed.
#define PICO_RATIO_MODE_NOT_SUPPORTED               0x00000047UL

// Aggregation was requested in rapid block mode.
#define PICO_RAPID_NOT_SUPPORT_AGGREGATION          0x00000048UL

// An invalid parameter was passed to <API>SetTriggerChannelProperties.
#define PICO_INVALID_TRIGGER_PROPERTY               0x00000049UL

// The driver was unable to contact the oscilloscope.
#define PICO_INTERFACE_NOT_CONNECTED                0x0000004AUL

// Resistance-measuring mode is not allowed in conjunction with the specified probe.
#define PICO_RESISTANCE_AND_PROBE_NOT_ALLOWED       0x0000004BUL

// The device was unexpectedly powered down.
#define PICO_POWER_FAILED                           0x0000004CUL

// A problem occurred in <API>SetSigGenBuiltIn or <API>SetSigGenArbitrary.
#define PICO_SIGGEN_WAVEFORM_SETUP_FAILED           0x0000004DUL

// FPGA not successfully set up.
#define PICO_FPGA_FAIL                              0x0000004EUL

#define PICO_POWER_MANAGER                          0x0000004FUL

// An impossible analog offset value was specified in <API>SetChannel.
#define PICO_INVALID_ANALOGUE_OFFSET                0x00000050UL

// There is an error within the device hardware.
#define PICO_PLL_LOCK_FAILED                        0x00000051UL

// There is an error within the device hardware.
#define PICO_ANALOG_BOARD                           0x00000052UL

// Unable to configure the signal generator.
#define PICO_CONFIG_FAIL_AWG                        0x00000053UL

// The FPGA cannot be initialized, so unit cannot be opened.
#define PICO_INITIALISE_FPGA                        0x00000054UL

// The frequency for the external clock is not within 15% of the nominal value.
#define PICO_EXTERNAL_FREQUENCY_INVALID             0x00000056UL

// The FPGA could not lock the clock signal.
#define PICO_CLOCK_CHANGE_ERROR                     0x00000057UL

// You are trying to configure the AUX input as both a trigger and a reference clock.
#define PICO_TRIGGER_AND_EXTERNAL_CLOCK_CLASH       0x00000058UL

// You are trying to congfigure the AUX input as both a pulse width qualifier and a reference clock.
#define PICO_PWQ_AND_EXTERNAL_CLOCK_CLASH           0x00000059UL

// The requested scaling file cannot be opened.
#define PICO_UNABLE_TO_OPEN_SCALING_FILE            0x0000005AUL

// The frequency of the memory is reporting incorrectly.
#define PICO_MEMORY_CLOCK_FREQUENCY                 0x0000005BUL

// The I2C that is being actioned is not responding to requests.
#define PICO_I2C_NOT_RESPONDING                     0x0000005CUL

// There are no captures available and therefore no data can be returned.
#define PICO_NO_CAPTURES_AVAILABLE                  0x0000005DUL

// The number of trigger channels is greater than 4,
// except for a PS4824 where 8 channels are allowed for rising/falling/rising_or_falling trigger directions.
#define PICO_TOO_MANY_TRIGGER_CHANNELS_IN_USE				0x0000005FUL

// When more than 4 trigger channels are set on a PS4824 and the direction is out of range.
#define PICO_INVALID_TRIGGER_DIRECTION							0x00000060UL

//  When more than 4 trigger channels are set and their trigger condition states are not <API>_CONDITION_TRUE.
#define PICO_INVALID_TRIGGER_STATES									0x00000061UL

// The capture mode the device is currently running in does not support the current request.
#define PICO_NOT_USED_IN_THIS_CAPTURE_MODE          0x0000005EUL

#define PICO_GET_DATA_ACTIVE                        0x00000103UL

// Codes 104 to 10B are used by the PT104 (USB) when connected via the Network Socket.

// The device is currently connected via the IP Network socket and thus the call made is not supported.
#define PICO_IP_NETWORKED                           0x00000104UL

// An incorrect IP address has been passed to the driver.
#define PICO_INVALID_IP_ADDRESS                     0x00000105UL

// The IP socket has failed.
#define PICO_IPSOCKET_FAILED                        0x00000106UL

// The IP socket has timed out.
#define PICO_IPSOCKET_TIMEDOUT                      0x00000107UL

// Failed to apply the requested settings.
#define PICO_SETTINGS_FAILED                        0x00000108UL

// The network connection has failed.
#define PICO_NETWORK_FAILED                         0x00000109UL

// Unable to load the WS2 DLL.
#define PICO_WS2_32_DLL_NOT_LOADED                  0x0000010AUL

// The specified IP port is invalid.
#define PICO_INVALID_IP_PORT                        0x0000010BUL

// The type of coupling requested is not supported on the opened device.
#define PICO_COUPLING_NOT_SUPPORTED                 0x0000010CUL

// Bandwidth limiting is not supported on the opened device.
#define PICO_BANDWIDTH_NOT_SUPPORTED                0x0000010DUL

// The value requested for the bandwidth limit is out of range.
#define PICO_INVALID_BANDWIDTH                      0x0000010EUL

// The arbitrary waveform generator is not supported by the opened device.
#define PICO_AWG_NOT_SUPPORTED                      0x0000010FUL

// Data has been requested with ETS mode set but run block has not been called, 
// or stop has been called.
#define PICO_ETS_NOT_RUNNING                        0x00000110UL

// White noise output is not supported on the opened device.
#define PICO_SIG_GEN_WHITENOISE_NOT_SUPPORTED       0x00000111UL

// The wave type requested is not supported by the opened device.
#define PICO_SIG_GEN_WAVETYPE_NOT_SUPPORTED         0x00000112UL

// The requested digital port number is out of range (MSOs only).
#define PICO_INVALID_DIGITAL_PORT                   0x00000113UL

// The digital channel is not in the range <API>_DIGITAL_CHANNEL0 to
// <API>_DIGITAL_CHANNEL15, the digital channels that are supported.
#define PICO_INVALID_DIGITAL_CHANNEL                0x00000114UL

// The digital trigger direction is not a valid trigger direction and should be equal
// in value to one of the <API>_DIGITAL_DIRECTION enumerations.
#define PICO_INVALID_DIGITAL_TRIGGER_DIRECTION      0x00000115UL

// Signal generator does not generate pseudo-random binary sequence.
#define	PICO_SIG_GEN_PRBS_NOT_SUPPORTED             0x00000116UL

// When a digital port is enabled, ETS sample mode is not available for use.
#define PICO_ETS_NOT_AVAILABLE_WITH_LOGIC_CHANNELS  0x00000117UL

#define PICO_WARNING_REPEAT_VALUE                   0x00000118UL

// 4-channel scopes only: The DC power supply is connected.
#define PICO_POWER_SUPPLY_CONNECTED                 0x00000119UL

// 4-channel scopes only: The DC power supply is not connected.
#define PICO_POWER_SUPPLY_NOT_CONNECTED             0x0000011AUL

// Incorrect power mode passed for current power source.
#define PICO_POWER_SUPPLY_REQUEST_INVALID           0x0000011BUL

// The supply voltage from the USB source is too low.
#define PICO_POWER_SUPPLY_UNDERVOLTAGE              0x0000011CUL

// The oscilloscope is in the process of capturing data.
#define PICO_CAPTURING_DATA                         0x0000011DUL

// A USB 3.0 device is connected to a non-USB 3.0 port.
#define PICO_USB3_0_DEVICE_NON_USB3_0_PORT          0x0000011EUL

// A function has been called that is not supported by the current device.
#define PICO_NOT_SUPPORTED_BY_THIS_DEVICE           0x0000011FUL

// The device resolution is invalid (out of range).
#define PICO_INVALID_DEVICE_RESOLUTION              0x00000120UL

// The number of channels that can be enabled is limited in 15 and 16-bit modes.
// (Flexible Resolution Oscilloscopes only)
#define PICO_INVALID_NUMBER_CHANNELS_FOR_RESOLUTION 0x00000121UL

// USB power not sufficient for all requested channels.
#define PICO_CHANNEL_DISABLED_DUE_TO_USB_POWERED    0x00000122UL

// The signal generator does not have a configurable DC offset.
#define PICO_SIGGEN_DC_VOLTAGE_NOT_CONFIGURABLE     0x00000123UL

// An attempt has been made to define pre-trigger delay without first enabling a trigger.
#define PICO_NO_TRIGGER_ENABLED_FOR_TRIGGER_IN_PRE_TRIG  0x00000124UL

// An attempt has been made to define pre-trigger delay without first arming a trigger.
#define PICO_TRIGGER_WITHIN_PRE_TRIG_NOT_ARMED           0x00000125UL

// Pre-trigger delay and post-trigger delay cannot be used at the same time.
#define PICO_TRIGGER_WITHIN_PRE_NOT_ALLOWED_WITH_DELAY   0x00000126UL

// The array index points to a nonexistent trigger.
#define PICO_TRIGGER_INDEX_UNAVAILABLE                   0x00000127UL

#define PICO_AWG_CLOCK_FREQUENCY									0x00000128UL

// There are more 4 analog channels with a trigger condition set.
#define PICO_TOO_MANY_CHANNELS_IN_USE							0x00000129UL

// The condition parameter is a null pointer.
#define PICO_NULL_CONDITIONS											0x0000012AUL

// There is more than one condition pertaining to the same channel.
#define PICO_DUPLICATE_CONDITION_SOURCE						0x0000012BUL	

// The parameter relating to condition information is out of range.
#define PICO_INVALID_CONDITION_INFO								0x0000012CUL	

// Reading the metadata has failed.
#define PICO_SETTINGS_READ_FAILED									0x0000012DUL

// Writing the metadata has failed.
#define PICO_SETTINGS_WRITE_FAILED								0x0000012EUL

// A parameter has a value out of the expected range.
#define PICO_ARGUMENT_OUT_OF_RANGE								0x0000012FUL

// The driver does not support the hardware variant connected.
#define PICO_HARDWARE_VERSION_NOT_SUPPORTED				0x00000130UL

// The driver does not support the digital hardware variant connected.
#define PICO_DIGITAL_HARDWARE_VERSION_NOT_SUPPORTED				0x00000131UL

// The driver does not support the analog hardware variant connected.
#define PICO_ANALOGUE_HARDWARE_VERSION_NOT_SUPPORTED				0x00000132UL

// Converting a channel's ADC value to resistance has failed.
#define PICO_UNABLE_TO_CONVERT_TO_RESISTANCE			0x00000133UL

// The channel is listed more than once in the function call.
#define PICO_DUPLICATED_CHANNEL										0x00000134UL

// The range cannot have resistance conversion applied.
#define PICO_INVALID_RESISTANCE_CONVERSION				0x00000135UL

// An invalid value is in the max buffer.
#define PICO_INVALID_VALUE_IN_MAX_BUFFER					0x00000136UL

// An invalid value is in the min buffer.
#define PICO_INVALID_VALUE_IN_MIN_BUFFER					0x00000137UL

// When calculating the frequency for phase conversion,  
// the frequency is greater than that supported by the current variant.
#define PICO_SIGGEN_FREQUENCY_OUT_OF_RANGE				0x00000138UL

// The device's EEPROM is corrupt. Contact Pico Technology support: https://www.picotech.com/tech-support.
#define PICO_EEPROM2_CORRUPT											0x00000139UL

// The EEPROM has failed.
#define PICO_EEPROM2_FAIL													0x0000013AUL

// The serial buffer is too small for the required information.
#define PICO_SERIAL_BUFFER_TOO_SMALL							0x0000013BUL

// The signal generator trigger and the external clock have both been set.
// This is not allowed.
#define PICO_SIGGEN_TRIGGER_AND_EXTERNAL_CLOCK_CLASH 0x0000013CUL

// The AUX trigger was enabled and the external clock has been enabled, 
// so the AUX has been automatically disabled.
#define PICO_WARNING_SIGGEN_AUXIO_TRIGGER_DISABLED 0x0000013DUL

// The AUX I/O was set as a scope trigger and is now being set as a signal generator
// gating trigger. This is not allowed.
#define PICO_SIGGEN_GATING_AUXIO_NOT_AVAILABLE	 	 0x00000013EUL

// The AUX I/O was set by the signal generator as a gating trigger and is now being set 
// as a scope trigger. This is not allowed.
#define PICO_SIGGEN_GATING_AUXIO_ENABLED				 	 0x00000013FUL

// A resource has failed to initialise 
#define PICO_RESOURCE_ERROR												 0x00000140UL

// The temperature type is out of range
#define PICO_TEMPERATURE_TYPE_INVALID							 0x000000141UL

// A requested temperature type is not supported on this device
#define PICO_TEMPERATURE_TYPE_NOT_SUPPORTED				 0x000000142UL

// A read/write to the device has timed out
#define PICO_TIMEOUT															 0x00000143UL

// The device cannot be connected correctly
#define PICO_DEVICE_NOT_FUNCTIONING							   0x00000144UL

// The driver has experienced an unknown error and is unable to recover from this error
#define PICO_INTERNAL_ERROR												 0x00000145UL

// Used when opening units via IP and more than multiple units have the same ip address
#define PICO_MULTIPLE_DEVICES_FOUND								 0x00000146UL

#define PICO_WARNING_NUMBER_OF_SEGMENTS_REDUCED 	 					0x00000147UL

// the calibration pin states argument is out of range
#define PICO_CAL_PINS_STATES											0x00000148UL

// the calibration pin frequency argument is out of range
#define PICO_CAL_PINS_FREQUENCY										0x00000149UL

// the calibration pin amplitude argument is out of range
#define PICO_CAL_PINS_AMPLITUDE										0x0000014AUL

// the calibration pin wavetype argument is out of range
#define PICO_CAL_PINS_WAVETYPE										0x0000014BUL

// the calibration pin offset argument is out of range
#define PICO_CAL_PINS_OFFSET											0x0000014CUL

// the probe's identity has a problem
#define PICO_PROBE_FAULT													0x0000014DUL

// the probe has not been identified
#define PICO_PROBE_IDENTITY_UNKNOWN								0x0000014EUL

// enabling the probe would cause the device to exceed the allowable current limit
#define PICO_PROBE_POWER_DC_POWER_SUPPLY_REQUIRED 0x0000014FUL

// the DC power supply is connected; enabling the probe would cause the device to exceed the
// allowable current limit
#define PICO_PROBE_NOT_POWERED_WITH_DC_POWER_SUPPLY 0x00000150UL

// failed to complete probe configuration
#define	PICO_PROBE_CONFIG_FAILURE									0x00000151UL

// failed to set the callback function, as currently in current callback function
#define	PICO_PROBE_INTERACTION_CALLBACK						0x00000152UL

// the probe has been verified but not know on this driver
#define	PICO_UNKNOWN_INTELLIGENT_PROBE						0x00000153UL

// the intelligent probe cannot be verified
#define	PICO_INTELLIGENT_PROBE_CORRUPT						0x00000154UL

// the callback is null, probe collection will only start when 
// first callback is a none null pointer
#define PICO_PROBE_COLLECTION_NOT_STARTED					0x00000155UL

// the current drawn by the probe(s) has exceeded the allowed limit
#define PICO_PROBE_POWER_CONSUMPTION_EXCEEDED    0x00000156UL

// the channel range limits have changed due to connecting or disconnecting a probe
// the channel has been enabled
#define PICO_WARNING_PROBE_CHANNEL_OUT_OF_SYNC 0x00000157UL

// The time stamp per waveform segment has been reset.
#define PICO_DEVICE_TIME_STAMP_RESET							0x01000000UL

// An internal erorr has occurred and a watchdog timer has been called.
#define PICO_WATCHDOGTIMER                        0x10000000UL

// The picoipp.dll has not been found.
#define PICO_IPP_NOT_FOUND                				0x10000001UL

// A function in the picoipp.dll does not exist.
#define PICO_IPP_NO_FUNCTION					            0x10000002UL

// The Pico IPP call has failed.
#define PICO_IPP_ERROR					                  0x10000003UL

// Shadow calibration is not available on this device.
#define PICO_SHADOW_CAL_NOT_AVAILABLE             0x10000004UL

// Shadow calibration is currently disabled.
#define PICO_SHADOW_CAL_DISABLED                  0x10000005UL

// Shadow calibration error has occurred.
#define PICO_SHADOW_CAL_ERROR                     0x10000006UL

// The shadow calibration is corrupt.
#define PICO_SHADOW_CAL_CORRUPT                   0x10000007UL

// the memory onboard the device has overflowed
#define PICO_DEVICE_MEMORY_OVERFLOW               0x10000008UL

#define PICO_RESERVED_1														0x11000000UL

typedef uint32_t PICO_STATUS;

typedef enum enPicoStringValue
{
	PICO_SV_MEMORY,
	PICO_SV_MEMORY_NO_OF_SEGMENTS,
	PICO_SV_MEMORY_MAX_SAMPLES,

	PICO_SV_NO_OF_CHANNELS,
	PICO_SV_ARRAY_OF_CHANNELS,
	PICO_SV_CHANNEL,
	PICO_SV_CHANNEL_NAME,
	PICO_SV_CHANNEL_RANGE,
	PICO_SV_CHANNEL_COUPLING,
	PICO_SV_CHANNEL_ENABLED,
	PICO_SV_CHANNEL_ANALOGUE_OFFSET,
	PICO_SV_CHANNEL_FILTER,

	PICO_SV_TRIGGER,
	PICO_SV_TRIGGER_AUXIO_OUTPUT_ENABLED,
	PICO_SV_TRIGGER_AUTO_TRIGGER_MILLISECONDS,

	PICO_SV_TRIGGER_PROPERTIES,
	PICO_SV_NO_OF_TRIGGER_PROPERTIES,

	PICO_SV_TRIGGER_PROPERTIES_CHANNEL,
	PICO_SV_TRIGGER_PROPERTIES_THRESHOLD_UPPER,
	PICO_SV_TRIGGER_PROPERTIES_THRESHOLD_UPPER_HYSTERESIS,
	PICO_SV_TRIGGER_PROPERTIES_THRESHOLD_LOWER,
	PICO_SV_TRIGGER_PROPERTIES_THRESHOLD_LOWER_HYSTERESIS,
	PICO_SV_TRIGGER_PROPERTIES_THRESHOLD_MODE,

	PICO_SV_TRIGGER_ARRAY_OF_BLOCK_CONDITIONS,
	PICO_SV_TRIGGER_NO_OF_BLOCK_CONDITIONS,
	PICO_SV_TRIGGER_CONDITIONS,
	PICO_SV_TRIGGER_NO_OF_CONDITIONS,
	PICO_SV_TRIGGER_CONDITION_SOURCE,
	PICO_SV_TRIGGER_CONDITION_STATE,

	PICO_SV_TRIGGER_DIRECTION,
	PICO_SV_TRIGGER_NO_OF_DIRECTIONS,
	PICO_SV_TRIGGER_DIRECTION_CHANNEL,
	PICO_SV_TRIGGER_DIRECTION_DIRECTION,

	PICO_SV_TRIGGER_DELAY,
	PICO_SV_TRIGGER_DELAY_MS,

	PICO_SV_FREQUENCY_COUNTER,
	PICO_SV_FREQUENCY_COUNTER_ENABLED,
	PICO_SV_FREQUENCY_COUNTER_CHANNEL,
	PICO_SV_FREQUENCY_COUNTER_RANGE,
	PICO_SV_FREQUENCY_COUNTER_TRESHOLDMAJOR,
	PICO_SV_FREQUENCY_COUNTER_TRESHOLDMINOR,

	PICO_SV_PULSE_WIDTH_PROPERTIES,
	PICO_SV_PULSE_WIDTH_PROPERTIES_DIRECTION,
	PICO_SV_PULSE_WIDTH_PROPERTIES_LOWER,
	PICO_SV_PULSE_WIDTH_PROPERTIES_UPPER,
	PICO_SV_PULSE_WIDTH_PROPERTIES_TYPE,

	PICO_SV_PULSE_WIDTH_ARRAY_OF_BLOCK_CONDITIONS,
	PICO_SV_PULSE_WIDTH_NO_OF_BLOCK_CONDITIONS,
	PICO_SV_PULSE_WIDTH_CONDITIONS,
	PICO_SV_PULSE_WIDTH_NO_OF_CONDITIONS,
	PICO_SV_PULSE_WIDTH_CONDITIONS_SOURCE,
	PICO_SV_PULSE_WIDTH_CONDITIONS_STATE,

	PICO_SV_SAMPLE_PROPERTIES,
	PICO_SV_SAMPLE_PROPERTIES_PRE_TRIGGER_SAMPLES,
	PICO_SV_SAMPLE_PROPERTIES_POST_TRIGGER_SAMPLES,
	PICO_SV_SAMPLE_PROPERTIES_TIMEBASE,
	PICO_SV_SAMPLE_PROPERTIES_NO_OF_CAPTURES,
	PICO_SV_SAMPLE_PROPERTIES_RESOLUTION,

	PICO_SV_SAMPLE_PROPERTIES_OVERLAPPED,
	PICO_SV_SAMPLE_PROPERTIES_OVERLAPPED_DOWN_SAMPLE_RATIO,
	PICO_SV_SAMPLE_PROPERTIES_OVERLAPPED_DOWN_SAMPLE_RATIO_MODE,
	PICO_SV_SAMPLE_PROPERTIES_OVERLAPPED_REQUERSTED_NO_OF_SAMPLES,
	PICO_SV_SAMPLE_PROPERTIES_OVERLAPPED_SEGMENT_INDEX_FROM,
	PICO_SV_SAMPLE_PROPERTIES_OVERLAPPED_SEGMENT_INDEX_TO,

	PICO_SV_SIGNAL_GENERATOR,
	PICO_SV_SIGNAL_GENERATOR_BUILT_IN,
	PICO_SV_SIGNAL_GENERATOR_BUILT_IN_WAVE_TYPE,
	PICO_SV_SIGNAL_GENERATOR_BUILT_IN_START_FREQUENCY,
	PICO_SV_SIGNAL_GENERATOR_BUILT_IN_STOP_FREQUENCY,
	PICO_SV_SIGNAL_GENERATOR_BUILT_IN_INCREMENT,
	PICO_SV_SIGNAL_GENERATOR_BUILT_IN_DWELL_TIME,
	PICO_SV_SIGNAL_GENERATOR_AWG,
	PICO_SV_SIGNAL_GENERATOR_AWG_START_DELTA_PHASE,
	PICO_SV_SIGNAL_GENERATOR_AWG_STOP_DELTA_PHASE,
	PICO_SV_SIGNAL_GENERATOR_AWG_DELTA_PHASE_INCREMENT,
	PICO_SV_SIGNAL_GENERATOR_AWG_DWELL_COUNT,
	PICO_SV_SIGNAL_GENERATOR_AWG_INDEX_MODE,
	PICO_SV_SIGNAL_GENERATOR_AWG_WAVEFORM_SIZE,
	PICO_SV_SIGNAL_GENERATOR_ARRAY_OF_AWG_WAVEFORM_VALUES,
	PICO_SV_SIGNAL_GENERATOR_OFFSET_VOLTAGE,
	PICO_SV_SIGNAL_GENERATOR_PK_TO_PK,
	PICO_SV_SIGNAL_GENERATOR_OPERATION,
	PICO_SV_SIGNAL_GENERATOR_SHOTS,
	PICO_SV_SIGNAL_GENERATOR_SWEEPS,
	PICO_SV_SIGNAL_GENERATOR_SWEEP_TYPE,
	PICO_SV_SIGNAL_GENERATOR_TRIGGER_TYPE,
	PICO_SV_SIGNAL_GENERATOR_TRIGGER_SOURCE,
	PICO_SV_SIGNAL_GENERATOR_EXT_IN_THRESHOLD,

	PICO_SV_ETS,
	PICO_SV_ETS_STATE,
	PICO_SV_ETS_CYCLE,
	PICO_SV_ETS_INTERLEAVE,
	PICO_SV_ETS_SAMPLE_TIME_PICOSECONDS

} PICO_STRING_VALUE;

#endif
