#include <iostream>
#include <dshow.h>
#include <atlbase.h>

#pragma comment( lib, "strmiids.lib" )

int main( int argc, char * argv[] )
{
	if ( argc != 2 )
	{
		std::cout << "Usage: " << argv[0] << " filename" << std::endl;
		return 1;
	}

	HRESULT hr;

	hr = ::CoInitialize( NULL );
	if FAILED( hr )
	{
		std::cout << "ERROR - Could not initialize COM library" << std::endl;
		return 1;
	}

	// フィルタグラフマネージャを作成し，インタフェースを問い合わせる．
	//CComPtr<IGraphBuilder> pGraph;
	CComQIPtr<IGraphBuilder, &IID_IGraphBuilder> pGraph;
	hr = pGraph.CoCreateInstance( CLSID_FilterGraph, NULL, CLSCTX_INPROC_SERVER );
	if FAILED( hr )
	{
		std::cout << "ERROR - Could not create the Filter Graph Manager." << std::endl;
		return 1;
	}

	//CComPtr<IMediaControl> pControl;
	CComQIPtr<IMediaControl, &IID_IMediaControl> pControl;
	hr = pGraph.QueryInterface( &pControl );

	//CComPtr<IMediaEven> pEvent;
	CComQIPtr<IMediaEvent, &IID_IMediaEvent> pEvent;
	hr = pGraph.QueryInterface( &pEvent );

	wchar_t filename[256];
	::MultiByteToWideChar( CP_ACP, MB_PRECOMPOSED, argv[1], -1, filename, 256 );

	// グラフを作成する．
	hr = pGraph->RenderFile( filename, NULL );
	if SUCCEEDED( hr )
	{
		// グラフを実行する。
		hr = pControl->Run();
		if SUCCEEDED( hr )
		{
			// 完了するまで待機する。
			long evCode;
			pEvent->WaitForCompletion( INFINITE, &evCode );

			// 注 : 実際のアプリケーションでは INFINITE を使用しないこと。
			// 無期限にブロックする場合がある。
		}
	}

	::CoUninitialize();

	return 0;
}
