import { getSettings } from "./platform";
// Small wrappers around Electron APIs used by several components.
const remote = window.require("@electron/remote");

export const isDev = !remote.app.isPackaged;

export function getSettings() {
  return getSettings();
}
