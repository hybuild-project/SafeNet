export const download = (url, logger = null) => {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", url, true);
    request.responseType = "arraybuffer";
    if (logger) {
      const [log, setState] = logger;
      request.onprogress = (e) => {
        const progress = (e.loaded / e.total) * 100;
        setState({ text: log, progress: progress.toFixed(2) });
      };
    }
    request.onload = function () {
      if (this.status >= 200 && this.status < 300) {
        if (!(request.response instanceof ArrayBuffer) || request.response.byteLength === 0) {
          reject(new Error(`Empty or invalid model response from ${url}`));
          return;
        }
        resolve(request.response);
      } else {
        reject(new Error(`HTTP ${this.status} ${request.statusText} loading ${url}`));
      }
    };
    request.onerror = function () {
      reject(new Error(`Network error loading ${url}`));
    };
    request.send();
  });
};
