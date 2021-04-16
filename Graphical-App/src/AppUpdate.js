/*************************************************************
 * Auto updating
 *************************************************************/

// const { app } = window.require("electron");
// import React, { Component } from "react";
const log = window.require("electron-log");
const { ipcRenderer } = window.require("electron");

function updateNotification(version) {
  let myNotification = new Notification("ML-SIM update available", {
    body:
      "ML-SIM version " +
      version +
      " is available. The download will start automatically.",
  });

  myNotification.onclick = () => {
    console.log("Notification clicked");
  };
}

export default function AppUpdate() {
  //   constructor(props) {
  //     super(props);
  //     this.state = {
  //       updateAvailable: false,
  //       downloadingUpdate: false,
  //       eta: null,
  //     };
  //   }

  ipcRenderer.on("update-available", (event, info) => {
    updateNotification(info.version);
    // this.setState({ updateAvailable: true });
  });
  ipcRenderer.on("update-not-available", (event, info) => {
    log.info("Update not available.");
  });
  ipcRenderer.on("error", (event, err) => {
    log.info("Error in auto-updater. " + err);
  });
  ipcRenderer.on("download-progress", (event, progressObj) => {
    // this.setState({ downloadingUpdate: true });
    let log_message = "Download speed: " + progressObj.bytesPerSecond;
    log_message = log_message + " - Downloaded " + progressObj.percent + "%";
    log_message =
      log_message +
      " (" +
      progressObj.transferred +
      "/" +
      progressObj.total +
      ")";
    log.info(log_message);
  });

  ipcRenderer.on("update-downloaded", (event, info) => {
    log.info("Update downloaded");
    // this.setState({ downloadingUpdate: false });
  });

  log.info("now checking from renderer");
  ipcRenderer.send("startUpdateService");

  // return (
  //   <div style={{width:100,backgroundColor:'blue',paddingTop:100}}>
  //     updateAvailable:{this.state.updateAvailable}, downloadingUpdate:
  //     {this.state.downloadingUpdate}, eta:null{" "}
  //   </div>
  // );
}
