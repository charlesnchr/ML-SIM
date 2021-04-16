const remote = window.require("electron").remote;
const shell = window.require("electron").shell;
const Store = window.require("electron-store");
const store = new Store();
const log = window.require("electron-log");
const Menu = remote.Menu;
const app = remote.app;
const process = window.require("process");
const isDev = window.require("electron-is-dev");

var sess = require("./sess.js");
// var pluginDict = require("./pluginDict.js");

let allPluginOptions = [{ name: "ML-SIM" }, { name: "ERNet" }];
let allowMultiplePlugins = false;
let previousPluginMenu;

let setPlugins = (plugin) => {
  let plugins = store.get("plugins");
  let checked = plugins.includes(plugin.name);
  if (checked) {
    plugins = plugins.filter((e) => e !== plugin.name);
    store.set("plugins", plugins);
  } else {
    if (allowMultiplePlugins) {
      plugins.push(plugin.name); // if more than one plugin should be allowed
    } else {
      plugins = [plugin.name];
    }
    WindowMenuRender(plugins);
    store.set("plugins", plugins);
  }
  sess.updateSidePanelPlugins(plugins);
};


let getPluginSubmenu = (checkedPlugins) => {
  let pluginSubmenu = [];
  allPluginOptions.forEach((plugin) => {
    pluginSubmenu.push({
      label: plugin.name,
      type: "checkbox",
      checked: checkedPlugins.includes(plugin.name),
      click: () => setPlugins(plugin)
    });
  });
  return pluginSubmenu;
}


/*************************************************************
 * Menu
 *************************************************************/
const WindowMenuRender = (checkedPlugins) => {
  let file_submenu = [];
  if (sess.settingsHandler != null) {
    file_submenu.push({
      label: "Settings",
      accelerator: "CmdOrCtrl+P",
      click: () => {
        sess.settingsHandler();
      },
    });
  }
  file_submenu.push(
    process.platform === "darwin" ? { role: "close" } : { role: "quit" }
  );

  const template = [
    // { role: 'appMenu' }
    ...(process.platform === "darwin"
      ? [
          {
            label: app.name,
            submenu: [
              { role: "about" },
              { type: "separator" },
              { role: "services" },
              { type: "separator" },
              { role: "hide" },
              { role: "hideothers" },
              { role: "unhide" },
              { type: "separator" },
              { role: "quit" },
            ],
          },
        ]
      : []),
    // { role: 'fileMenu' }
    {
      label: "File",
      submenu: file_submenu,
    },
    // { role: 'viewMenu' }
    {
      label: "View",
      submenu: [
        { role: "reload" },
        // { role: 'toggledevtools' },
        ...(isDev
          ? [
              { role: "toggledevtools" },
              {
                label: "Relaunch",
                accelerator: "CmdOrCtrl+Q",
                click: () => {
                  app.relaunch();
                  app.exit(0);
                },
              },
            ]
          : []),
        { type: "separator" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    // { role: 'windowMenu' }
    {
      label: "Window",
      submenu: [
        { role: "minimize" },
        { role: "zoom" },
        ...(process.platform === "darwin"
          ? [
              { type: "separator" },
              { role: "front" },
              { type: "separator" },
              { role: "window" },
            ]
          : [{ role: "close" }]),
      ],
    },
    {
      role: "help",
      submenu: [
        {
          label: "Support",
          click: async () => {
            await shell.openExternal(sess.baseurl + "/support");
          },
        },
        {
          label: "About",
          accelerator: "CmdOrCtrl+H",
          click: () => {
            sess.aboutHandler();
          },
        },
      ],
    },
  ];

  let pluginSubmenu;

  if(checkedPlugins == null) { // init
    pluginSubmenu = getPluginSubmenu(store.get("plugins"));
  } else {
    pluginSubmenu = getPluginSubmenu(checkedPlugins)
  }

  template.push({
    label: "Plugins",
    submenu: pluginSubmenu,
  });

  previousPluginMenu = pluginSubmenu;

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

export default WindowMenuRender;
