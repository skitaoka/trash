import java.awt.AWTKeyStroke;
import java.awt.FlowLayout;
import java.awt.KeyboardFocusManager;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.util.HashSet;
import java.util.Set;

import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JTextField;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;

// cf. https://sites.google.com/site/shin1ogawa/java/swing/enter-focus-traversal
public class FocusTraverse extends JFrame {

  public FocusTraverse() {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setLayout(new FlowLayout());
    JTextField text1 = new JTextField("text1");
    JTextField text2 = new JTextField("text2");
    JButton button = new JButton("Button");

    add(text1);
    add(text2);
    add(button);

    addEnterPolicy(text1);
    addEnterPolicy(text2);
    addEnterPolicy(button);

    pack();
    setVisible(true);
  }

  private void addEnterPolicy(JComponent comp) {
    //次へのフォーカス設定
    {
      Set<AWTKeyStroke> keystrokes = new HashSet<AWTKeyStroke>();
      keystrokes.addAll(comp.getFocusTraversalKeys(KeyboardFocusManager.FORWARD_TRAVERSAL_KEYS));

      //ENTERを追加
      keystrokes.add(KeyStroke.getAWTKeyStroke(KeyEvent.VK_ENTER, 0));
      comp.setFocusTraversalKeys(KeyboardFocusManager.FORWARD_TRAVERSAL_KEYS, keystrokes);
    }

    //前へのフォーカス設定
    {
      Set<AWTKeyStroke> keystrokes = new HashSet<AWTKeyStroke>();
      keystrokes.addAll(comp.getFocusTraversalKeys(KeyboardFocusManager.BACKWARD_TRAVERSAL_KEYS));

      // Shift+Enterを追加
      keystrokes.add(KeyStroke.getAWTKeyStroke(KeyEvent.VK_ENTER, InputEvent.SHIFT_MASK));
      comp.setFocusTraversalKeys(KeyboardFocusManager.BACKWARD_TRAVERSAL_KEYS, keystrokes);
    }
  }

  public static void main(String[] args) {
    SwingUtilities.invokeLater(new Runnable() {
      public void run() {
        new FocusTraverse();
      }
    });
  }
}
