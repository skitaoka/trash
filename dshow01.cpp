#define NOMINMAX
#define STRICT
#include <windows.h>
#include <tchar.h>
#include <iostream>
#include <dshow.h>
#include <atlbase.h>

//#pragma comment( lib, "strmiids.lib" )

int main( int argc, char * argv[] )
{
  if ( argc != 2 )
  {
    std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
    ::Sleep( 500 );
    return 1;
  }

  HRESULT hr;

  hr = ::CoInitialize( NULL );
  if FAILED( hr )
  {
    std::cout << "ERROR - Could not initialize COM library" << std::endl;
    ::Sleep( 500 );
    return 1;
  }

  // フィルタグラフマネージャを作成し，インタフェースを問い合わせる．
  IGraphBuilder * pGraph;
  hr = ::CoCreateInstance(
    CLSID_FilterGraph,
    NULL,
    CLSCTX_INPROC_SERVER,
    IID_IGraphBuilder,
    reinterpret_cast<LPVOID *>( &pGraph ) );
  if FAILED( hr )
  {
    std::cout << "ERROR - Could not create the Filter Graph Manager." << std::endl;
    ::Sleep( 500 );
    return 1;
  }

  IMediaControl * pControl;
  hr = pGraph->QueryInterface( IID_IMediaControl, reinterpret_cast<LPVOID *>( &pControl ) );

  IMediaEvent * pEvent;
  hr = pGraph->QueryInterface( IID_IMediaEvent, reinterpret_cast<LPVOID *>( &pEvent ) );

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

  pControl->Stop();
  pControl->Release();
  pEvent->Release();
  pGraph->Release();

  ::CoUninitialize();

  return 0;
}
