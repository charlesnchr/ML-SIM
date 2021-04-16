module.exports = {
    baseurl: "https://ML-SIM.com",
    imgsize : 250, // default width
    headerHeight : 70,
    topOffset : 100,
    sidePanelWidth : 230,
    pageYOffset : 0, // stored scroll position (triggered when clearing results)
    width : window.innerWidth,
    height : window.innerHeight,

    desktop_fingerprint : "-1",
    serverActive : false,

    // functions
    updateGeometry : null,
    updateSidePanel : null,
    selectedFilepaths: [],
    removeResults : null,
    readFolders : null,
    removeSelection : null,
    isCalcFeaturesFinished : null,
    toggleSelectedFilepath : null,
    resort : false,
    sortBy : "filename",

    displayedFolders : null,
    dirsizes : null,

    filepaths : null,
    filepaths_hash : "",
    showingResult : false,
    resultsDepth : 0, // number of sequential searches
    resultsBuffer : [],
    thumbJobs : [],
    thumbQueue : [],
    thumbdict : {},
    excludePattern : "",
    includePattern : "",
}