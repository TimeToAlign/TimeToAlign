/*
 * Version menu and outdated-version banner for the TimeToAlign! documentation.
 *
 * The site is published as one folder per version beside this script:
 * `<X.Y.Z>/` per release, `dev/` for the main branch and `stable/` for a
 * copy of the newest release, listed in `versions.json`. `publish_site.py`
 * copies this file to the root of the published tree and adds a
 * `<script src=".../versions.js" defer>` tag to every HTML page, so the
 * script works with any site generator.
 *
 * The script (a) adds a small fixed-position version menu that opens the
 * same page in another version, or that version's start page when the page
 * does not exist there, and (b) shows a banner on the pages of the
 * development version and of releases older than the newest one.
 */
(function () {
  "use strict";

  var script = document.currentScript;
  if (!script || document.getElementById("tta-version-menu")) {
    return;
  }
  var rootUrl = new URL(".", script.src);
  var rootPath = rootUrl.pathname;

  function currentLocation() {
    var path = window.location.pathname;
    if (path.indexOf(rootPath) !== 0) {
      return null;
    }
    var rest = path.slice(rootPath.length);
    var slash = rest.indexOf("/");
    if (slash < 0) {
      return null;
    }
    return { folder: rest.slice(0, slash), page: rest.slice(slash + 1) };
  }

  function pageExists(url) {
    return fetch(url, { method: "HEAD" })
      .then(function (response) {
        return response.ok;
      })
      .catch(function () {
        return false;
      });
  }

  function switchTo(entry, page) {
    var start = new URL(entry.path, rootUrl);
    var target = new URL(page, start);
    pageExists(target.href).then(function (exists) {
      window.location.href = exists ? target.href + window.location.hash : start.href;
    });
  }

  function pageColors() {
    var style = window.getComputedStyle(document.body);
    var background = style.backgroundColor;
    if (!background || background === "transparent" || background === "rgba(0, 0, 0, 0)") {
      background = "Canvas";
    }
    return { background: background, text: style.color || "CanvasText" };
  }

  function addStyle(colors) {
    var css =
      "#tta-version-menu{position:fixed;right:12px;bottom:12px;z-index:1050;" +
      "display:flex;align-items:center;gap:6px;padding:4px 8px;border-radius:6px;" +
      "font:13px/1.4 system-ui,sans-serif;opacity:.85;" +
      "border:1px solid currentColor;color-scheme:light dark;" +
      "background:" + colors.background + ";color:" + colors.text + "}" +
      "#tta-version-menu:hover,#tta-version-menu:focus-within{opacity:1}" +
      "#tta-version-menu select{font:inherit;color:inherit;background:inherit;" +
      "border:1px solid currentColor;border-radius:4px;padding:1px 4px}" +
      "#tta-version-banner{padding:8px 16px;text-align:center;" +
      "font:14px/1.5 system-ui,sans-serif;background:#fff3cd;color:#664d03;" +
      "border-bottom:1px solid #ffe69c}" +
      "#tta-version-banner a{color:inherit;font-weight:600;text-decoration:underline}" +
      "@media (prefers-color-scheme: dark){#tta-version-banner{background:#332701;" +
      "color:#ffda6a;border-bottom-color:#664d03}}";
    var element = document.createElement("style");
    element.textContent = css;
    document.head.appendChild(element);
  }

  function addMenu(versions, here) {
    var menu = document.createElement("div");
    menu.id = "tta-version-menu";
    var label = document.createElement("label");
    label.htmlFor = "tta-version-select";
    label.textContent = "Version";
    var select = document.createElement("select");
    select.id = "tta-version-select";
    versions.forEach(function (entry, index) {
      var option = document.createElement("option");
      option.value = String(index);
      option.textContent = entry.title || entry.version;
      option.selected = entry.path === here.folder + "/";
      select.appendChild(option);
    });
    select.addEventListener("change", function () {
      switchTo(versions[Number(select.value)], here.page);
    });
    menu.appendChild(label);
    menu.appendChild(select);
    document.body.appendChild(menu);
  }

  function addBanner(message, stable) {
    var banner = document.createElement("div");
    banner.id = "tta-version-banner";
    banner.setAttribute("role", "note");
    banner.appendChild(document.createTextNode(message + " "));
    var link = document.createElement("a");
    link.href = new URL(stable.path, rootUrl).href;
    link.textContent = "Go to the documentation of the current release.";
    banner.appendChild(link);
    document.body.insertBefore(banner, document.body.firstChild);
  }

  function bannerMessage(versions, here) {
    var releases = versions.filter(function (entry) {
      return entry.release === true;
    });
    if (here.folder === "dev") {
      return "You are reading the documentation of the development version.";
    }
    if (releases.length === 0 || here.folder === "stable") {
      return null;
    }
    var newest = releases[0].version;
    var isOlderRelease = releases.some(function (entry, index) {
      return index > 0 && entry.version === here.folder;
    });
    if (!isOlderRelease) {
      return null;
    }
    return (
      "You are reading the documentation of version " + here.folder + ". " +
      "The current release is " + newest + "."
    );
  }

  function start(versions) {
    var here = currentLocation();
    if (!here || !Array.isArray(versions)) {
      return;
    }
    addStyle(pageColors());
    addMenu(versions, here);
    var stable = versions.filter(function (entry) {
      return entry.version === "stable";
    })[0];
    var message = bannerMessage(versions, here);
    if (message && stable) {
      addBanner(message, stable);
    }
  }

  fetch(new URL("versions.json", rootUrl).href, { cache: "no-cache" })
    .then(function (response) {
      return response.ok ? response.json() : null;
    })
    .then(start)
    .catch(function () {
      /* Without versions.json the page stays as it is. */
    });
})();
