import java.net.HttpURLConnection;
import java.net.URL;
import java.net.MalformedURLException;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.io.IOException;

import java.util.zip.GZIPInputStream;

public final class Talk2ch {
	private String host_ = "ex21.2ch.net";
	private String board_ = "voiceactor";
	private String thread_ = "1171964460";

	public void start() {
		//URL url = getSubjectURL();
		URL url = getDatURL();
		if (url == null) {
			return;
		}
		System.out.println( url );

		try {
			// 接続
			HttpURLConnection connection = (HttpURLConnection)url.openConnection();
			try {
				connection.setRequestMethod("GET");
				connection.setRequestProperty("Accept-Encoding", "gzip");
				connection.setRequestProperty("Host", host_);
				connection.setRequestProperty("User-Agent", "Monazilla/1.00");
				connection.setRequestProperty("Connection", "close");
				connection.connect();

				if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
					// 失敗
					System.err.println( "接続に失敗しました: " + connection.getResponseCode() );
					return;
				}

				loadThreads( connection );
			} finally {
				connection.disconnect();
			}
		} catch (IOException e) {
		}
	}

	private void loadThreads(HttpURLConnection connection) throws IOException {
		InputStream is = connection.getInputStream();

		// 圧縮されていれば一段かます
		String encoding = connection.getHeaderField("Content-Encoding");
		if ("gzip".equals(encoding)) {
			System.out.println("圧縮されてる");
			is = new GZIPInputStream( is );
		}

		BufferedReader in = new BufferedReader(new InputStreamReader(is));
		try {
			String line;
			while ((line = in.readLine()) != null) {
				System.out.println( line );
			}
		} finally {
			in.close();
		}
	}

	private URL getSubjectURL() {
		try {
			return new URL( "http://" + host_ + '/' + board_ + "/subject.txt" );
		} catch ( MalformedURLException e ) {
			return null;
		}
	}

	public URL getDatURL() {
		try {
			return new URL( "http://" + host_ + '/' + board_ + "/dat/" + thread_ + ".dat" );
		} catch (MalformedURLException e) {
			return null;
		}
	}

	public static void main(String [] args) {
		new Talk2ch().start();
	}
}

/*
DAT差分を取得

要求ヘッダに次の項目を追加する

If-Modified-Since: リモートDATの最終更新時刻(値は応答ヘッダのLast-Modifiedを調べる)
Range: bytes=ローカルDATのファイルサイズ-

ファイルサイズの単位はバイト。ローカルdatは2chのdatと完全に同じであることが前提条件。
Windowsデフォルトの改行は\r\n、2chデフォルトの改行は\nなので、1バイト×レス数のファイルサイズ不一致、という罠に嵌りやすいので注意。
ローカルdatが2chと違う仕様なら、受信したデータのサイズ(解凍後)を累積記録などして、それを使用する。

If-Modified-SinceとDateヘッダの値は、RFC1123形式で表される時刻。例えば、Fri, 30 Mar 2001 22:35:45 +0900

DATが更新されていれば206 HTTP_PARTIAL_CONTENTが返ってきて、差分データを取得できる。
更新(新着)なしならば304 HTTP_NOT_MODIFIEDが返ってくる

あぼーんがあった場合、ファイルサイズ不一致により、データが取得できません。(ローカルDAT＞リモートDAT)
返ってくるステータスは416 HTTP_RANGE_NOT_SATISFIABLEになります。

参考までに、あぼーんをより確実に検出するために、次のような工夫をして差分取得しているブラウザもあるようです。

If-Modified-SinceにリモートDATの更新時刻、rangeに取得済みDATのサイズマイナス1を指定して要求する。
新着なしなら304 HTTP_NOT_MODIFIEDが返ってきて、データは送信されない
新着あり、つまりContent-lengthが2以上の場合は、受信データの先頭を確認して\nだったら正常、\n以外だったらあぼーん。
新着なし＋あぼーんがあった時は416。(ローカルDAT＞リモートDAT)
*/
