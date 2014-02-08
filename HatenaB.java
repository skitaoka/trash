import java.util.Date;
import java.util.Random;
import java.util.TimeZone;

import java.text.SimpleDateFormat;

import java.io.ByteArrayOutputStream;
import java.io.UnsupportedEncodingException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;

import java.net.HttpURLConnection;
import java.net.URL;

import java.security.SecureRandom;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/*
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
*/

import com.sun.mail.util.BASE64EncoderStream; // javamail 1.4.1 の mailapi.jar

public final class HatenaB {
	public static void main(String[] args) throws IOException {
		if (postHatenaBookmark(args[0], args[1], args[2], args[3])) {
			System.out.println("成功");
		} else {
			System.out.println("失敗");
		}
	}

	/// はてなブックマークに投稿する
	private static boolean postHatenaBookmark(String username, String password, String link, String comment) {
		try {
			HttpURLConnection connection = (HttpURLConnection)new URL("http://b.hatena.ne.jp/atom/post").openConnection();
			connection.setDoOutput(true);
			connection.setRequestMethod("POST");
			connection.setRequestProperty("Content-Type", "application/atom+xml");
			connection.setRequestProperty("X-WSSE", getWsseHeaderValue(username, password));
			connection.connect();
			try {
				PrintStream out = new PrintStream(connection.getOutputStream(), false, "UTF-8");
				try {
					out.println("<?xml version=\"1.0\" encoding=\"utf-8\"?>");
					out.println("<entry xmlns=\"http://purl.org/atom/ns#\">");
					out.println("  <title>dummy</title>");
					out.println("  <link rel=\"related\" type=\"text/html\" href=\"" + link + "\" />");
					out.println("  <summary type=\"text/plain\">" + comment + "</summary>");
					out.println("</entry>");
					out.flush();
				} finally {
					out.close();
				}

				if (connection.getResponseCode() != 201) {
					return false;
				}

				//DocumentBuilderFactory dbfactory = DocumentBuilderFactory.newInstance();
				//DocumentBuilder builder = dbfactory.newDocumentBuilder();
				//return builder.parse( connection.getInputStream() );
			} finally {
				connection.disconnect();
			}
		} catch (Exception e) {
			return false;
		}

		return true;
	}

	/// WSSE認証用のヘッダ文字列を生成する
	private static String getWsseHeaderValue(String username, String password) {
		try {
			// セキュリティ用のランダムな数を生成する
			byte[] bytesNonce = new byte[16];
			SecureRandom.getInstance("SHA1PRNG").nextBytes( bytesNonce );
			String nonce = toStringBase64( bytesNonce );

			// ランダムトークンを生成した日時(UTC, ISO-8601)を生成する
			String created = getDateTimeUTC();
			byte[] bytesCreated = created.getBytes("UTF-8");

			// パスワードをバイト列に直す
			byte[] bytesPassword = password.getBytes("UTF-8");

			// WSSE認証用のPasswordDigestを生成する
			final String digest;
			{
				MessageDigest md = MessageDigest.getInstance("SHA-1");
				md.update( bytesNonce );
				md.update( bytesCreated );
				md.update( bytesPassword );
				digest = toStringBase64(md.digest());
			}

			// こいつを X-WSSE: に仕込む
			return "UsernameToken Username=\"" + username +
					"\", PasswordDigest=\"" + digest +
					"\", Nonce=\"" + nonce +
					"\", Created=\"" + created + '"';
		} catch (UnsupportedEncodingException e) {
			throw new Error( "UTF-8エンコーディングがサポートされていないJVMシステムです", e );
		} catch (NoSuchAlgorithmException e) {
			throw new Error( "SHA-1ハッシュ生成がサポートされていないJVMシステムです", e );
		}
	}

	/// ISO-8601でUTCの日時の文字列を取得する
	private static String getDateTimeUTC() {
		SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");
		formatter.setTimeZone( TimeZone.getTimeZone("GMT") );
		return formatter.format( new Date() );
	}

	/// Base64エンコードで文字列にする
	private static String toStringBase64(byte[] data) {
		return new String(BASE64EncoderStream.encode(data));
	}

	/// バイト列を16進文字列に変換する
	private static String toString(byte[] data) {
		StringBuilder buf = new StringBuilder();
		for (int i = 0, size = data.length; i < size; ++i) {
			buf.append(Character.forDigit((data[i]>>>4)&0xF, 16));
			buf.append(Character.forDigit((data[i]    )&0xF, 16));
		}
		return buf.toString();
	}
}
