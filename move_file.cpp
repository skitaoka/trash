// �t�@�C���̍폜�ƈړ����A�g�~�b�N�ɏ�������.

#ifdef _WIN32
#include <windows.h> // DeleteFile, MoveFileEx �̂���
#else
#include <cstdio>    // rename �̂���
#include <unistd.h>  // unlink �̂���
#endif

// �폜
// ���������Ƃ��� 0 ���Ԃ�B
// ���s�����Ƃ��� -1 ���Ԃ�B
int remove_file(char const * const path)
{
#ifdef _WIN32
  return ::DeleteFile(path) ? 0 : -1;
#else
  return ::unlink(path);
#endif
}

// �ړ�
// �t�@�C������ path_old ���� path_new �֕ς���B
// path_new �����łɑ��݂��Ă�ꍇ�́A�㏑������B
// ���������Ƃ��� 0 ���Ԃ�B
// ���s�����Ƃ��� -1 ���Ԃ�B
int rename_file(char const * const path_old, char const * const path_new)
{
#ifdef _WIN32
  return ::MoveFileEx(path_old, path_new, MOVEFILE_REPLACE_EXISTING) ? 0 : -1;
#else
  return ::rename(path_old, path_new);
#endif
}

/*
Java �ŃA�g�~�b�N�ɏ㏑�����l�C�������i�� 1.7 �ȍ~�łȂ��ƂȂ��B
** Files.move(source, target, StandardCopyOption.ATOMIC_MOVE)�B

* Windows �ł� Java �� FileReader �Ƃ� FileInputStream �Ńt�@�C�����J���Ă���ƁA��L�̑��삪���s����B
* Linux �ł͐�������B�t�@�C�����͂͌p�����ĉ\�ɂȂ��Ă���i�n�[�h�����N�\���̂������j�B
** Windows �ł� Vista �ȍ~�Ńn�[�h�����N������悤�ɂȂ����̂ŁA����𗘗p���悤�I
*** Files.createLink(newLink, existingFile); // �Ή����Ă���̂� Java 1.7 �ȍ~�c�c
*** �ǂ����Ă��Ƃ����ꍇ�́A�R�}���h�𒼐ڂ�т������B
**** Windows: mklink /h newLink existingFile
***** Vista �ȍ~�ŉ\�AXP �͕s�A����ɊǗ��Ҍ������K�v�����B
**** Linux: ln existingFile newLink
*/
