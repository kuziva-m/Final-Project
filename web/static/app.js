// Progress feedback for slow actions. Without JavaScript the forms still
// submit normally; this only adds the visible "something is happening" state.

(function () {
  "use strict";

  // --- Upload: real upload progress, then a timer while the server reads -----
  var form = document.getElementById("upload");
  var busy = document.getElementById("busy");

  if (form && busy && window.XMLHttpRequest && window.FormData) {
    var titleEl = document.getElementById("busy-title");
    var fill = document.getElementById("busy-bar-fill");
    var detail = document.getElementById("busy-detail");
    var retry = document.getElementById("busy-retry");
    var timer = null;

    var mb = function (bytes) { return (bytes / 1048576).toFixed(1) + " MB"; };
    var clock = function (secs) {
      var s = Math.floor(secs);
      return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
    };
    var setState = function (state) {
      busy.classList.remove("is-uploading", "is-reading", "is-failed");
      busy.classList.add("is-" + state);
    };
    var fail = function (message) {
      clearInterval(timer);
      setState("failed");
      titleEl.textContent = "Something went wrong";
      detail.textContent = message;
      retry.hidden = false;
      retry.focus();
    };

    retry.addEventListener("click", function () {
      busy.hidden = true;
      retry.hidden = true;
      document.body.style.overflow = "";
    });

    form.addEventListener("submit", function (event) {
      var input = form.querySelector('input[type="file"]');
      if (!input || !input.files.length) return; // let the browser's "required" message show

      event.preventDefault();
      var file = input.files[0];
      var isPdf = /\.pdf$/i.test(file.name) || file.type === "application/pdf";
      var xhr = new XMLHttpRequest();
      clearInterval(timer);

      busy.hidden = false;
      retry.hidden = true;
      document.body.style.overflow = "hidden";
      setState("uploading");
      fill.style.width = "0%";
      titleEl.textContent = "Uploading… 0%";
      detail.textContent = file.name + " · " + mb(file.size);

      // Step 1: send the file. /upload only saves it and answers at once, so
      // "upload finished" is exact (browsers report the end of an upload only
      // when the server starts answering).
      xhr.upload.onprogress = function (e) {
        if (!e.lengthComputable) return;
        var pct = Math.min(100, Math.round((e.loaded / e.total) * 100));
        fill.style.width = pct + "%";
        titleEl.textContent = "Uploading… " + pct + "%";
        detail.textContent = mb(e.loaded) + " of " + mb(e.total);
      };
      xhr.onload = function () {
        var reply = null;
        try { reply = JSON.parse(xhr.responseText); } catch (err) { /* not JSON */ }
        if (xhr.status !== 200 || !reply || !reply.token) {
          fail((reply && reply.error) || "The upload failed (" + xhr.status + "). Try again.");
          return;
        }
        read(reply.token, isPdf);
      };
      xhr.onerror = function () {
        fail("The connection dropped while uploading. Check your connection and try again.");
      };
      xhr.open("POST", form.getAttribute("data-upload-url"));
      var data = new FormData();
      data.append("image", file);
      xhr.send(data);
    });

    // Step 2: ask the server to read the saved file, with a running timer.
    var read = function (token, isPdf) {
      setState("reading");
      fill.style.width = "";
      titleEl.textContent = isPdf ? "Reading your pages…" : "Reading your ledger…";
      var started = Date.now();
      var tick = function () {
        var hint = isPdf ? "Large PDFs can take a few minutes. " : "This usually takes under a minute. ";
        detail.textContent = hint + "Elapsed " + clock((Date.now() - started) / 1000);
      };
      tick();
      timer = setInterval(tick, 1000);

      var xhr = new XMLHttpRequest();
      xhr.onload = function () {
        var reply = null;
        try { reply = JSON.parse(xhr.responseText); } catch (err) { /* not JSON */ }
        if (xhr.status === 200 && reply && reply.review_url) {
          // A normal page load, so the review page works (and refreshes) like any other.
          titleEl.textContent = "Opening your table…";
          window.location.href = reply.review_url;
          return;
        }
        clearInterval(timer);
        if (reply && reply.error) {
          fail(reply.error);
        } else {
          fail("The server hit an error while reading this file (" + xhr.status + "). Try again, or try a smaller file.");
        }
      };
      xhr.onerror = function () {
        fail("The connection dropped or the server took too long to answer. Check your connection and try again.");
      };
      var body = new FormData();
      body.append("token", token);
      xhr.open("POST", form.action);
      xhr.send(body);
    };
  }

  // --- Save: show it's saving, and stop a double click saving twice --------
  var rowsForm = document.querySelector("form.rows-form");
  if (rowsForm) {
    rowsForm.addEventListener("submit", function () {
      var button = rowsForm.querySelector('button[type="submit"]');
      if (!button) return;
      // Disable after the submit has been dispatched, so the form still sends.
      setTimeout(function () {
        button.disabled = true;
        button.textContent = "Saving…";
      }, 0);
    });
  }
})();
