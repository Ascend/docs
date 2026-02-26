var _hmt = _hmt || [];
(function() {
  var hm = document.createElement("script");
  hm.src = "https://hm.baidu.com/hm.js?102ee21b68fd0e419aa031b4b9645cb5";
  var s = document.getElementsByTagName("script")[0]; 
  s.parentNode.insertBefore(hm, s);
})();

document.addEventListener('DOMContentLoaded', function () {
  var aside = document.querySelector('.wy-breadcrumbs-aside');
  if (!aside) return;
  var anchors = aside.querySelectorAll('a');
  if (anchors.length) {
    anchors.forEach(function (a) {
      a.setAttribute('href', 'https://github.com/Ascend/docs');
      a.setAttribute('target', '_blank');
      a.textContent = '访问 GitHub 仓库';
    });
  }
});

