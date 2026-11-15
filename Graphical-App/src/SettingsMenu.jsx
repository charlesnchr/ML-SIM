import React from "react";
import "./App.css";
import Button from "@material-ui/core/Button";
import Box from "@material-ui/core/Box";
import Typography from "@material-ui/core/Typography";
import ButtonGroup from "@material-ui/core/ButtonGroup";
import Dialog from "@material-ui/core/Dialog";
import DialogTitle from "@material-ui/core/DialogTitle";
import DialogActions from "@material-ui/core/DialogActions";
import DialogContent from "@material-ui/core/DialogContent";
import DialogContentText from "@material-ui/core/DialogContentText";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import InputLabel from "@material-ui/core/InputLabel";
import Switch from "@material-ui/core/Switch";
import InputBase from "@material-ui/core/InputBase";
import Grid from "@material-ui/core/Grid";
import AppBar from "@material-ui/core/AppBar";
import Tabs from "@material-ui/core/Tabs";
import Tab from "@material-ui/core/Tab";

import SpeedIcon from "@material-ui/icons/Speed";
import PermMediaIcon from "@material-ui/icons/PermMedia";
import ColorLensIcon from "@material-ui/icons/ColorLens";

function a11yProps(index) {
  return {
    id: `scrollable-force-tab-${index}`,
    "aria-controls": `scrollable-force-tabpanel-${index}`,
  };
}

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <Typography
      component="div"
      role="tabpanel"
      hidden={value !== index}
      id={`scrollable-auto-tabpanel-${index}`}
      aria-labelledby={`scrollable-auto-tab-${index}`}
      {...other}
    >
      {value === index && <Box p={3}>{children}</Box>}
    </Typography>
  );
}

const Store = window.require("electron-store");
const store = new Store();

export default function SettingsMenu(props) {
  const { onClose, open, exportdir, cachedir } = props;

  const [value, setValue] = React.useState(0);
  const [allowGPU, set_allowGPU] = React.useState(store.get("allowGPU"));
  const [scanSubdir, set_scanSubdir] = React.useState(store.get("scanSubdir"));

  const handle_allowGPU = (event, newValue) => {
    set_allowGPU(event.target.checked);
    store.set("allowGPU", event.target.checked);
  };

  const handle_scanSubdir = (event, newValue) => {
    set_scanSubdir(event.target.checked);
    store.set("scanSubdir", event.target.checked);
  };

  const handleChange = (event, newValue) => {
    setValue(newValue);
  };

  const handleClose = () => {
    onClose();
  };

  const openFolder = (folder) => {
    window.require("child_process").exec('explorer.exe "' + folder + '"');
  };

  const gridItemStyle = { minHeight: 250 };

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
        Configure settings for ML-SIM
      </DialogTitle>
      <DialogContent style={{ padding: "0 60px 30px 60px" }}>
        <DialogContentText style={{ marginBottom: 30 }}>
          Changes from the default settings may result in worse performance and
          unexpected behavior. Proceed with caution.
        </DialogContentText>
        <form
          noValidate
          style={{
            display: "flex",
            flexDirection: "column",
            margin: "auto",
            width: "fit-content",
          }}
        >
          <Grid container direction="column" spacing={3}>
            <AppBar
              position="static"
              color="default"
              style={{ marginBottom: 20 }}
            >
              <Tabs
                value={value}
                onChange={handleChange}
                indicatorColor="primary"
                textColor="primary"
                variant="scrollable"
                scrollButtons="auto"
                aria-label="scrollable auto tabs example"
              >
                <Tab
                  label="Directories"
                  icon={<PermMediaIcon />}
                  {...a11yProps(0)}
                />
                <Tab
                  label="Performance"
                  icon={<SpeedIcon />}
                  {...a11yProps(1)}
                />
                <Tab
                  label="Interface"
                  icon={<ColorLensIcon />}
                  {...a11yProps(1)}
                  disabled
                />
              </Tabs>
            </AppBar>
            <TabPanel value={value} index={0}>
              <Grid item style={gridItemStyle}>
                <Grid container direction="column" spacing={2}>
                  {/* <Grid item>
                            <Box fontWeight='bold'>Directories</Box>
                            <hr style={{marginBottom:10}}/>
                        </Grid> */}
                  <Grid item>
                    <Box fontSize="small">
                      Below directories are automatically generated - they may
                      not yet exist.
                    </Box>
                  </Grid>
                  <Grid item style={{ width: "100%" }}>
                    <InputLabel>Root directory for export</InputLabel>
                    <InputBase
                      style={{
                        backgroundColor: "rgb(220,220,220)",
                        borderRadius: 5,
                        padding: 5,
                        fontSize: 13,
                        width: "100%",
                      }}
                      value={exportdir}
                    />
                  </Grid>
                  <Grid item align="center">
                    <ButtonGroup
                      color="default"
                      aria-label="text default button group"
                    >
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={openFolder.bind(this, exportdir)}
                      >
                        Open folder
                      </Button>
                      <Button variant="outlined" size="small" disabled>
                        Choose new
                      </Button>
                    </ButtonGroup>
                    <Button
                      variant="outlined"
                      color="secondary"
                      size="small"
                      style={{ marginLeft: 20 }}
                    >
                      Clean directory
                    </Button>
                  </Grid>
                  <Grid item style={{ width: "100%", marginTop: 20 }}>
                    <InputLabel>Library for indexed images</InputLabel>
                    <InputBase
                      style={{
                        backgroundColor: "rgb(220,220,220)",
                        borderRadius: 5,
                        padding: 5,
                        fontSize: 13,
                        width: "100%",
                      }}
                      value={cachedir}
                    />
                  </Grid>
                  <Grid item align="center">
                    <ButtonGroup
                      color="default"
                      aria-label="text default button group"
                    >
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={openFolder.bind(this, cachedir)}
                      >
                        Open folder
                      </Button>
                      <Button variant="outlined" size="small" disabled>
                        Choose new
                      </Button>
                    </ButtonGroup>
                    <Button
                      variant="outlined"
                      color="secondary"
                      size="small"
                      style={{ marginLeft: 20 }}
                    >
                      Clean directory
                    </Button>
                  </Grid>
                  <Grid item>
                    <Box fontSize="small">
                      The functionality to change the default directories is not
                      yet implemented. 
                    </Box>
                  </Grid>
                </Grid>
              </Grid>
            </TabPanel>
            <TabPanel value={value} index={1}>
              <Grid item style={gridItemStyle}>
                <Grid container direction="column" spacing={2}>
                  {/* <Grid item>
                            <Box fontWeight='bold'>Performance</Box>
                            <hr/>
                        </Grid> */}
                  <Grid item>
                    <FormControlLabel
                      label="Include subdirectories when scanning for images"
                      control={
                        <Switch
                          checked={scanSubdir}
                          onChange={handle_scanSubdir}
                        />
                      }
                    />
                  </Grid>
                  <Grid item>
                    <FormControlLabel
                      label="Enable hardware acceleration (for Nvidia graphics cards)"
                      control={
                        <Switch checked={allowGPU} onChange={handle_allowGPU} />
                      }
                    />
                  </Grid>
                  <Grid item>
                    <Box fontSize="small">
                      {" "}
                      Below options will be available in the next version of
                      ML-SIM.{" "}
                    </Box>
                  </Grid>
                  <Grid item>
                    <FormControlLabel
                      label="Update library when an indexed folder changes"
                      control={<Switch checked={false} disabled />}
                    />
                  </Grid>
                  <Grid item>
                    <Box fontSize="small">
                      If you would like to be able to control other settings,
                      such as limiting CPU utilisation or memory consumption,
                      let us know at{" "}
                      <a href="mailto:cnc39@cam.ac.uk">cnc39@cam.ac.uk</a>{" "}
                    </Box>
                  </Grid>
                </Grid>
              </Grid>
            </TabPanel>
          </Grid>
        </form>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} color="primary" variant="outlined">
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}
