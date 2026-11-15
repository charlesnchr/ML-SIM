// import React from 'react';
// import InputBase from '@material-ui/core/InputBase';
// import { fade, makeStyles } from '@material-ui/core/styles';
// import SearchIcon from '@material-ui/icons/Search';
// import { submit } from './App.js';

// const useStyles = makeStyles(theme => ({
//   search: {
//     position: 'relative',
//     borderRadius: theme.shape.borderRadius,
//     backgroundColor: fade(theme.palette.common.white, 0.15),
//     '&:hover': {
//       backgroundColor: fade(theme.palette.common.white, 0.25),
//     },
//     marginLeft: 0,
//     width: '100%',
//     [theme.breakpoints.up('sm')]: {
//       marginLeft: 0,
//       width: 'auto',
//     },
//   },
//   searchIcon: {
//     width: theme.spacing(7),
//     height: '100%',
//     position: 'absolute',
//     pointerEvents: 'none',
//     display: 'flex',
//     alignItems: 'center',
//     justifyContent: 'center',
//   },
//   inputRoot: {
//     color: 'inherit',
//   },
//   inputInput: {
//     padding: theme.spacing(1, 1, 1, 7),
//     transition: theme.transitions.create('width'),
//     width: '100%',
//     fontSize: '0.85em',
//     [theme.breakpoints.up('sm')]: {
//       width: 100,
//       '&:focus': {
//         width: 100,
//       },
//     },
//   },
// }));



// export default function SearchBar() {
//     const classes = useStyles();
  

//     return (
//             <div className={classes.search}>
//               <div className={classes.searchIcon}>
//                 <SearchIcon />
//               </div>
//               <InputBase size='small'
//                 placeholder="Text search…"
//                 classes={{
//                   root: classes.inputRoot,
//                   input: classes.inputInput,
//                 }}
//                 inputProps={{ 'aria-label': 'search' }}
//                 onKeyPress={submit.bind(this)}
//                 id='searchval'
//               />
//             </div>
//     );
//   }