document.addEventListener("DOMContentLoaded", function () {
  // ---- Upload drag & drop + preview ----
  const dropArea = document.getElementById("uploadDrop");
  const fileInput = document.getElementById("image");
  const previewImg = document.getElementById("previewImg");
  const previewWrap = document.getElementById("previewWrap");
  const uploadForm = document.getElementById("uploadForm");
  const overlay = document.getElementById("spinnerOverlay");

  function showPreview(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
      previewWrap.classList.remove("d-none");
    };
    reader.readAsDataURL(file);
  }

  if (dropArea && fileInput) {
    dropArea.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach((evt) =>
      dropArea.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.add("dragover");
      })
    );
    ["dragleave", "drop"].forEach((evt) =>
      dropArea.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.remove("dragover");
      })
    );
    dropArea.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length) {
        fileInput.files = files;
        showPreview(files[0]);
      }
    });
    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) showPreview(fileInput.files[0]);
    });
  }

  if (uploadForm) {
    uploadForm.addEventListener("submit", function () {
      if (fileInput.files && fileInput.files.length) {
        overlay.style.display = "flex";
      }
    });
  }

  // ---- Animated probability bars on result page ----
  document.querySelectorAll(".animated-bar").forEach((bar) => {
    const target = bar.getAttribute("data-width");
    setTimeout(() => {
      bar.style.width = target + "%";
    }, 150);
  });

  // ---- Confidence ring (simple conic-gradient based) ----
  const ring = document.getElementById("confidenceRing");
  if (ring) {
    const pct = parseFloat(ring.getAttribute("data-pct"));
    ring.style.background = `conic-gradient(#2fb8a6 ${pct * 3.6}deg, #e6edf3 0deg)`;
  }

  // ---- Delete confirmation ----
  document.querySelectorAll(".confirm-delete").forEach((form) => {
    form.addEventListener("submit", function (e) {
      if (!confirm("Delete this analysis record permanently?")) {
        e.preventDefault();
      }
    });
  });

  // ---- Dismiss alerts automatically ----
  document.querySelectorAll(".alert").forEach((alertEl) => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alertEl);
      if (bsAlert) bsAlert.close();
    }, 6000);
  });
});