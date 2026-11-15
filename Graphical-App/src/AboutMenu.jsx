import React from "react";
import "./App.css";
import Button from "@material-ui/core/Button";
import Dialog from "@material-ui/core/Dialog";
import DialogTitle from "@material-ui/core/DialogTitle";
import DialogActions from "@material-ui/core/DialogActions";
import DialogContent from "@material-ui/core/DialogContent";
import DialogContentText from "@material-ui/core/DialogContentText";

export default function AboutMenu(props) {
  const { onClose, open, version } = props;


  const handleClose = () => {
    onClose();
  };

  return (
    <Dialog
      onClose={handleClose}
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
        <DialogContentText style={{ marginBottom: 30 }}>
          Copyright © 2020-2026 Charles N. Christensen. Released under the GNU General Public License v3.0.
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
