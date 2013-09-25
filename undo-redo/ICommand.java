public interface ICommand {
  void invoke();
  void undo();
  void redo();
}
