(function () {
  function isQuickstartPage() {
    var pathname = window.location.pathname.toLowerCase();
    return pathname.indexOf('onnxruntime') !== -1 && pathname.indexOf('quick_start') !== -1;
  }

  document.addEventListener('DOMContentLoaded', function () {
    if (!isQuickstartPage()) {
      return;
    }

    document.body.classList.add('onnxruntime-quickstart-page');
  });
})();
