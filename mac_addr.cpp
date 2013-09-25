#define VC_EXTRALEAN 1
#define NOMINMAX 1
#define WIN32_LEAN_AND_MEAN 1
#define _USE_MATH_DEFINES 1
#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#define _CRT_SECURE_NO_DEPRECATE 1
#define _NO_DEBUG_HEAP 1

#include <iostream>
#include <algorithm>
#include <iterator>
#include <memory>

#include <Windows.h>
#include <Nb30.h>

#pragma comment(lib, "Netapi32")

int main()
{
	NCB ncb;

	// NIC 数の取得
	LANA_ENUM ncbes;
	std::memset(&ncb, 0, sizeof(ncb));
	ncb.ncb_command = NCBENUM;
	ncb.ncb_buffer  = reinterpret_cast<PUCHAR>(&ncbes);
	ncb.ncb_length  = sizeof(ncbes);

	if (UCHAR const bResult = ::Netbios(&ncb)) {
		std::cerr << "Error: " << bResult << std::endl;
		return bResult; // error
	}

	std::cout << "num_ncbes = " << static_cast<int>(ncbes.length) << std::endl;
	for (int i = 0, size = ncbes.length; i < size; ++i) {
		// NIC をリセット
		std::memset(&ncb, 0, sizeof(ncb));
		ncb.ncb_command  = NCBRESET;
		ncb.ncb_lana_num = ncbes.lana[i];
		if (UCHAR const bResult = ::Netbios(&ncb)) {
			std::cerr << "Error: " << bResult << std::endl;
			continue;
		}
		
		// アダプタの状態を取得
		ADAPTER_STATUS  adapt;
		std::memset(&ncb, 0, sizeof(ncb));
		ncb.ncb_command  = NCBASTAT;
		ncb.ncb_lana_num = ncbes.lana[i];
		ncb.ncb_buffer   = reinterpret_cast<PUCHAR>(&adapt);
		ncb.ncb_length   = sizeof(adapt);
		std::string const callname("*               ");
		std::copy(callname.cbegin(), callname.cend(), ncb.ncb_callname);
		if (UCHAR const bResult = ::Netbios(&ncb)) {
			std::cerr << "Error: " << bResult << std::endl;
			continue;
		}

#if 0
		std::printf( "MAC:  %02X%02X%02X%02X%02X%02X\n",
				adapt.adapter_address[0],
				adapt.adapter_address[1],
				adapt.adapter_address[2],
				adapt.adapter_address[3],
				adapt.adapter_address[4],
				adapt.adapter_address[5]);
#else
		typedef unsigned long long int mac_addr_t;
		mac_addr_t mac_addr = 0;
		for (int i = 0; i < 6; ++i) {
			mac_addr = (mac_addr << 8) | adapt.adapter_address[i];
		}
		std::printf( "MAC>  %012X\n", mac_addr);
#endif
	}

	return 0;
}
