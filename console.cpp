int main(int argv,char ** argc)
{
    CONSOLE_SCREEN_INFO c_clore = FOREGROUND_RED;
    CONSOLE_SCREEN_INFO std_clore = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE ;
    CONSOLE_SCREEN_INFO ch_clore =　FOREGROUND_BLUE、FOREGROUND_GREEN;
    HANDLE h_stdout = GetStdHandle( STD_OUTPUT_HANDLE );
    
    SetConsoleTextAttribute( h_stdout, c_clore);
    printf("Hello !!");
    
    SetConsoleTextAttribute( h_stdout, std_clore);
    printf("http://etherpad.com/nYiyifu0BB で編集中")
    
    return 0;
}


/*
FOREGROUND_BLUE、FOREGROUND_GREEN、FOREGROUND_RED、FOREGROUND_INTENSITY、BACKGROUND_BLUE、BACKGROUND_GREEN、BACKGROUND_RED、BACKGROUND_INTENSITY を自由に組み合わせて指定することができます。
たとえば、次の組み合わせを指定すると、テキストは白、背景は黒になります。 
FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE 
*/
