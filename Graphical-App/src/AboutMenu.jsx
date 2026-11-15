import React from "react";
import "./App.css";
import Button from "@material-ui/core/Button";
import Dialog from "@material-ui/core/Dialog";
import DialogTitle from "@material-ui/core/DialogTitle";
import DialogActions from "@material-ui/core/DialogActions";
import DialogContent from "@material-ui/core/DialogContent";
import DialogContentText from "@material-ui/core/DialogContentText";

const Store = window.require("electron-store");
const store = new Store();

export default function AboutMenu(props) {
  const { onClose, open, version } = props;

  const [pdist_version, set_pdist_version] = React.useState("N/A");

  const get_pdist_version = (event) => {
    let pdist_version = store.get("pdist_version");
    if (pdist_version) set_pdist_version(pdist_version);
  };

  const handleClose = () => {
    onClose();
  };

  return (
    <Dialog
      onClose={handleClose}
      onEnter={get_pdist_version}
      aria-labelledby="simple-dialog-title"
      open={open}
    >
      <DialogTitle
        style={{ padding: "40px 60px 30px 40px" }}
        id="simple-dialog-title"
      >
        About ML-SIM
      </DialogTitle>
      <DialogContent style={{ padding: "0 60px 30px 60px" }}>
        <p>Your version of ML-SIM is {version}</p>
        <p>Your version of ML-SIM Engine is {pdist_version}</p>
        <DialogContentText style={{ marginBottom: 30 }}>
          Copyright © 2020-2025 All Rights Reserved by Charles N. Christensen(MIT License)
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} color="primary" variant="outlined">
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}
