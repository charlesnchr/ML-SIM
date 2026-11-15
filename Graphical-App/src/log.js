// Renderer-side logging. Messages go to the DevTools console.
const log = {
  info: (...args) => console.debug(...args),
  warn: (...args) => console.warn(...args),
  error: (...args) => console.error(...args),
};

export default log;
