class base_WindowClass
{
protected:
  LRESULT WindowProc(HWND, UINT, WPARAM, LPARAM);
};

template <class base_Class>
class WindowClass: public base_Class
{
public:
  WindowClass(): base_Class() {}
  WindowClass(base_Class const & src): base_Class(src) {}

  ATOM RegisterClassEx(
    UINT style, int cbClsExtra, int cbWndExtra, HINSTANCE hinst,
    HICON hIcon, HCURSOR hCursor, HBRUSH hbrBackground,
    LPCTSTR lpszMenuName, LPCTSTR lpszClassName, HICON hIconSm)
  {
    WNDCLASSEX wcex =
      {sizeof(WNDCLASSEX),style,initWindowProc,cbClsExtra,cbWndExtra,hinst,
       hIcon,hCursor,hbrBackground,lpszMenuName,lpszClassName,hIconSm};
    return ::RegisterClassEx(&wcex);
  }

  HWND CreateWindowEx(
    DWORD dwExStyle,LPCTSTR lpClassName,LPCTSTR lpWindowName,DWORD dwStyle,
    int x,int y,int nWidth,int nHeight,HWND hWndParent,HMENU hMenu,HINSTANCE hinst)
  {
    return ::CreateWindowEx(
        dwExStyle,lpClassName,lpWindowName,dwStyle,
        x,y,nWidth,nHeight,hWndParent,hMenu,hinst,this);
  }

private:
//初期化専用
  static LRESULT CALLBACK initWindowProc(HWND hwnd,UINT uMsg,WPARAM wParam,LPARAM lParam)
  {
    if (WM_CREATE == uMsg) {
      WindowClass *this_ = (WindowClass *)((LPCREATESTRUCT)lParam)->lpCreateParams;
      ::SetWindowLongPtr(hwnd,GWL_USERDATA,(LONG_PTR)this_);
      ::SetWindowLongPtr(hwnd,GWL_WNDPROC,(LONG_PTR)staticWindowProc);
      return this_->WindowProc(hwnd,uMsg,wParam,lParam);
    }
    return ::DefWindowProc(hwnd,uMsg,wParam,lParam);
  }

//間接参照
  static LRESULT CALLBACK staticWindowProc(HWND hwnd,UINT uMsg,WPARAM wParam,LPARAM lParam)
  {
    return ((WindowClass *)::GetWindowLongPtr(hwnd,GWL_USERDATA))->WindowProc(hwnd,uMsg,wParam,lParam);
  }
};
