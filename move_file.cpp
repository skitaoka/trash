// ファイルの削除と移動をアトミックに処理する.

#ifdef _WIN32
#include <windows.h> // DeleteFile, MoveFileEx のため
#else
#include <cstdio>    // rename のため
#include <unistd.h>  // unlink のため
#endif

// 削除
// 成功したときに 0 が返る。
// 失敗したときは -1 が返る。
int remove_file(char const * const path)
{
#ifdef _WIN32
  return ::DeleteFile(path) ? 0 : -1;
#else
  return ::unlink(path);
#endif
}

// 移動
// ファイル名を path_old から path_new へ変える。
// path_new がすでに存在してる場合は、上書きする。
// 成功したときに 0 が返る。
// 失敗したときは -1 が返る。
int rename_file(char const * const path_old, char const * const path_new)
{
#ifdef _WIN32
  return ::MoveFileEx(path_old, path_new, MOVEFILE_REPLACE_EXISTING) ? 0 : -1;
#else
  return ::rename(path_old, path_new);
#endif
}

/*
Java でアトミックに上書きリネイムする手段は 1.7 以降でないとない。
** Files.move(source, target, StandardCopyOption.ATOMIC_MOVE)。

* Windows では Java で FileReader とか FileInputStream でファイルを開いていると、上記の操作が失敗する。
* Linux では成功する。ファイル入力は継続して可能になっている（ハードリンク構造のおかげ）。
** Windows でも Vista 以降でハードリンクを作れるようになったので、これを利用しよう！
*** Files.createLink(newLink, existingFile); // 対応しているのは Java 1.7 以降……
*** どうしてもという場合は、コマンドを直接よびだすか。
**** Windows: mklink /h newLink existingFile
***** Vista 以降で可能、XP は不可、さらに管理者権限が必要かも。
**** Linux: ln existingFile newLink
*/
