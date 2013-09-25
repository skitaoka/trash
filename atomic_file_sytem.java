import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;

final class atomic_file_sytem {
  public static void main(final String[] args) throws IOException, InterruptedException {
    final Path file = Paths.get(args[0]);
    final Path temp = Paths.get(args[0] + ".link.tmp");
    final Path link = Paths.get(args[0] + ".link");

    // テンポラリのリンクを作る
    if (Files.exists(temp)) {
      try {
        Files.delete(temp);
      } catch (final NoSuchFileException e) {
        // ファイルがないのは問題ない
      }
    }
    Files.createLink(temp, file);

    // 正しいリンクに直す（アトミック）。
    Files.move(temp, link, StandardCopyOption.ATOMIC_MOVE);

    // 一度、テンポラリのリンクを作るのは、リンクがすでにあったとき、createLink をアトミックにできないため。

    try {
      final BufferedReader in = Files.newBufferedReader(link, StandardCharsets.UTF_8);
      try {
        for (String line = in.readLine(); line != null; line = in.readLine()) {
          System.out.println(line);
          Thread.sleep(1000L);
        }
      } finally {
        in.close();
      }
    } finally {
      Files.delete(link);
    }
  }
}