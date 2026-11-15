// Small wrappers around Electron APIs used by several components.
const { shell } = window.require("electron");
const remote = window.require("@electron/remote");

export const isDev = !remote.app.isPackaged;

export function getSettings() {
  return remote.getGlobal("settings");
}

// Open a folder in the system file manager.
export function openFolder(folder) {
  shell.openPath(folder);
}

// Open the system file manager with the given file selected.
export function revealFile(filepath) {
  shell.showItemInFolder(filepath);
}
