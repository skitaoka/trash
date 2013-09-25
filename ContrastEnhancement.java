import java.io.File;
import java.io.IOException;

import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

// Yxy色空間でのコントラスト改善
final class ContrastEnhancement {

  public static void main(String[] args) {
    if (args.length > 1) {
      System.err.println("Usage: java CE file.jpg");
      return;
    }
    File input = new File(args[0]);
    if (!input.exists() || !input.isFile()) {
      System.err.printf("File not found: %s\n", args[0]);
      return;
    }
    try {
      // input
      BufferedImage image = ImageIO.read(input);
      final int w = image.getWidth();
      final int h = image.getHeight();
      final int n = w * h;
      final double[] X = new double[n];
      final double[] Y = new double[n];
      final double[] Z = new double[n];
      final double[] cx = new double[n];
      final double[] cy = new double[n];
      for (int i = 0, y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x, ++i) {
          final int rgb = image.getRGB(x, y);

          // sRGB
          double R = ((rgb >> 16) & 0xFF) / 255.0;
          double G = ((rgb >>  8) & 0xFF) / 255.0;
          double B = ((rgb      ) & 0xFF) / 255.0;

          // RGB to XYZ
          X[i] = 0.412453 * R + 0.357580 * G + 0.180423 * B; if (X[i] < 0) { X[i] = 0; } else if (X[i] > 1) { X[i] = 1; }
          Y[i] = 0.212671 * R + 0.715160 * G + 0.072169 * B; if (Y[i] < 0) { Y[i] = 0; } else if (Y[i] > 1) { Y[i] = 1; }
          Z[i] = 0.019334 * R + 0.119193 * G + 0.950227 * B; if (Z[i] < 0) { Z[i] = 0; } else if (Z[i] > 1) { Z[i] = 1; }

          // XYZ to Yxy
          final double S = X[i] + Y[i] + Z[i];
          if (S > 0) {
            cx[i] = X[i] / S;
            cy[i] = Y[i] / S;
          } else {
            cx[i] = 0;
            cy[i] = 0;
          }
        }
      }
      
      // error histgram
      double[] H = new double[256+1];
      for (int i = 0, y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x, ++i) {
          double e = 0;
          int a = 0;
          for (int dy = -1; dy <= 1; ++dy) {
            final int yy = y + dy;
            if (yy <  0) continue;
            if (yy >= h) break;
            for (int dx = -1; dx <= 1; ++dx) {
              if ((dy == 0) && (dx == 0)) continue;
              final int xx = x + dx;
              if (xx <  0) continue;
              if (xx >= w) break;
              
              final int j = yy * w + xx;
              final double ex = X[i] - X[j];
              final double ey = Y[i] - Y[j];
              final double ez = Z[i] - Z[j];
              e += Math.sqrt(ex * ex + ey * ey + ez * ez);
              ++a;
            }
          }
          H[(int)(Y[i] * 255)+1] += e / a; // accumulate error
        }
      }
      for (int i = 0, m = H.length - 1; i < m; ++i) {
        H[i+1] += H[i];
      }
      for (int i = 0, m = H.length - 1; i < m; ++i) {
        H[i+1] /= H[m];
      }
      
      // output
      for (int i = 0, y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x, ++i) {
          final double Ly = H[(int)(Y[i] * 255)];
          final double Lx;
          final double Lz;
          if (cy[i] > 0) {
            final double Lyy = Ly / cy[i];
            Lx = Lyy * cx[i];
            Lz = Lyy * (1 - cx[i] - cy[i]);
          } else {
            Lx = 0;
            Lz = 0;
          }
          double R =  3.240479 * Lx - 1.537150 * Ly - 0.498535 * Lz; if (R < 0) { R = 0; } else if (R > 1) { R = 1; }
          double G = -0.969256 * Lx + 1.875991 * Ly + 0.041556 * Lz; if (G < 0) { G = 0; } else if (G > 1) { G = 1; }
          double B =  0.055648 * Lx - 0.204043 * Ly + 1.057311 * Lz; if (B < 0) { B = 0; } else if (B > 1) { B = 1; }

          final int rgb = 0xff000000
            | ((int)(R * 255) << 16)
            | ((int)(G * 255) <<  8)
            | ((int)(B * 255)      );
          image.setRGB(x, y, rgb);
        }
      }
      
      File output = new File(args[0] + ".jpg");
      ImageIO.write(image, "jpg", output);
    } catch (IOException e) {
    }
  }
}
