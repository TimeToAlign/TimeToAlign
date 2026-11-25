/*
 * Version menu and outdated-version banner of a documentation site.
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
  if (!script || document.getElementById("docs-version-menu")) {
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

  function luminance(color) {
    var match = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/.exec(color || "");
    if (!match) {
      return null;
    }
    var channels = [match[1], match[2], match[3]].map(function (value) {
      var c = Number(value) / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
  }

  function isTransparent(color) {
    return (
      !color ||
      color === "transparent" ||
      /^rgba\(.*,\s*0(\.0+)?\s*\)$/.test(color) ||
      /\/\s*0(\.0+)?\s*\)$/.test(color)
    );
  }

  // The colours of the page as the site draws it (its theme may differ from
  // the operating system's colour scheme): the background of the body, else
  // of the root element, else the browser's default canvas, read as light.
  function pageColors() {
    var body = window.getComputedStyle(document.body);
    var background = body.backgroundColor;
    if (isTransparent(background)) {
      background = window.getComputedStyle(document.documentElement).backgroundColor;
    }
    if (isTransparent(background)) {
      background = "Canvas";
    }
    var level = luminance(background);
    return {
      background: background,
      text: body.color || "CanvasText",
      dark: level !== null && level < 0.5
    };
  }

  function addStyle(colors) {
    var banner = colors.dark
      ? { background: "#332701", text: "#ffda6a", border: "#664d03" }
      : { background: "#fff3cd", text: "#664d03", border: "#ffe69c" };
    var scheme = colors.dark ? "dark" : "light";
    var css =
      "#docs-version-menu{position:fixed;right:12px;bottom:12px;z-index:1050;" +
      "display:flex;align-items:center;gap:6px;margin:0;padding:4px 8px;" +
      "border-radius:6px;font:13px/1.4 system-ui,sans-serif;opacity:.85;" +
      "border:1px solid currentColor;color-scheme:" + scheme + ";" +
      "background:" + colors.background + ";color:" + colors.text + "}" +
      "#docs-version-menu:hover,#docs-version-menu:focus-within{opacity:1}" +
      "#docs-version-menu select,#docs-version-menu button{font:inherit;" +
      "color:inherit;background:inherit;border:1px solid currentColor;" +
      "border-radius:4px;padding:1px 4px;margin:0}" +
      "#docs-version-menu button{cursor:pointer}" +
      "#docs-version-banner{padding:8px 16px;text-align:center;" +
      "font:14px/1.5 system-ui,sans-serif;color-scheme:" + scheme + ";" +
      "background:" + banner.background + ";color:" + banner.text + ";" +
      "border-bottom:1px solid " + banner.border + "}" +
      "#docs-version-banner a{color:inherit;font-weight:600;text-decoration:underline}";
    var element = document.createElement("style");
    element.textContent = css;
    document.head.appendChild(element);
  }

  // A form, so that choosing a version in the list (with the arrow keys, for
  // instance) navigates only once the reader activates "Go".
  function addMenu(versions, here) {
    var menu = document.createElement("form");
    menu.id = "docs-version-menu";
    menu.setAttribute("aria-label", "Documentation version");
    var label = document.createElement("label");
    label.htmlFor = "docs-version-select";
    label.textContent = "Version";
    var select = document.createElement("select");
    select.id = "docs-version-select";
    versions.forEach(function (entry, index) {
      var option = document.createElement("option");
      option.value = String(index);
      option.textContent = entry.title || entry.version;
      option.selected = entry.path === here.folder + "/";
      select.appendChild(option);
    });
    var go = document.createElement("button");
    go.type = "submit";
    go.textContent = "Go";
    menu.addEventListener("submit", function (event) {
      event.preventDefault();
      switchTo(versions[Number(select.value)], here.page);
    });
    menu.appendChild(label);
    menu.appendChild(select);
    menu.appendChild(go);
    document.body.appendChild(menu);
  }

  function addBanner(message, stable) {
    var banner = document.createElement("div");
    banner.id = "docs-version-banner";
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
