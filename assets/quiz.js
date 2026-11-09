// Quiz and predict-then-reveal widgets, shared by every lesson.
//
// Quiz markup:
//   <div class="quiz" data-answer="1">          (0-based index of the right option)
//     <p class="q">Question</p>
//     <div class="opts"><button>A</button><button>B</button>…</div>
//     <p class="why">Explanation, shown after the first click.</p>
//   </div>
// A group of quizzes inside <section data-quiz-set> gets a running score in its .quiz-score.
//
// Predict markup:
//   <div class="predict" data-key="unique-key">
//     <textarea></textarea><button>Lock in</button>
//     <div class="reveal">…</div>
//   </div>

(function () {
  function store(op, key, value) {
    try {
      return op === "get" ? localStorage.getItem(key) : localStorage.setItem(key, value);
    } catch (_) {
      return null;
    }
  }

  function scoreSet(set) {
    var out = set.querySelector(".quiz-score");
    if (!out) return;
    var all = set.querySelectorAll(".quiz");
    var done = set.querySelectorAll(".quiz.done");
    var right = set.querySelectorAll(".quiz.done.got-it");
    out.textContent = done.length === all.length
      ? right.length + " / " + all.length + " on first try."
      : "";
  }

  document.querySelectorAll(".quiz").forEach(function (quiz) {
    var answer = Number(quiz.dataset.answer);
    var buttons = quiz.querySelectorAll(".opts button");
    buttons.forEach(function (btn, i) {
      btn.addEventListener("click", function () {
        if (quiz.classList.contains("done")) return;
        quiz.classList.add("done");
        if (i === answer) quiz.classList.add("got-it");
        else btn.classList.add("wrong");
        buttons[answer].classList.add("right");
        buttons.forEach(function (b) { b.disabled = true; });
        var set = quiz.closest("[data-quiz-set]");
        if (set) scoreSet(set);
      });
    });
  });

  document.querySelectorAll(".predict").forEach(function (box) {
    var key = "predict:" + box.dataset.key;
    var area = box.querySelector("textarea");
    var saved = store("get", key);
    if (saved) {
      area.value = saved;
      box.classList.add("locked");
    }
    box.querySelector("button").addEventListener("click", function () {
      if (!area.value.trim()) {
        area.focus();
        return;
      }
      store("set", key, area.value);
      box.classList.add("locked");
      area.readOnly = true;
    });
    if (saved) area.readOnly = true;
  });
})();
