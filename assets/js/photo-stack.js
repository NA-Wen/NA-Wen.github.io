(() => {
  "use strict";

  document.querySelectorAll("[data-photo-stack]").forEach((stack) => {
    const cards = Array.from(stack.querySelectorAll("[data-photo-card]"));
    const caption = stack.querySelector("[data-photo-caption]");
    if (!cards.length || !caption) return;

    let order = cards.slice();

    const bringForward = (card) => {
      if (order[0] === card) return;
      order = [card, ...order.filter((item) => item !== card)];
      order.forEach((item, depth) => {
        item.dataset.depth = String(depth);
        item.style.setProperty("--depth", String(depth));
        item.setAttribute("aria-pressed", String(depth === 0));
      });
      caption.textContent = card.dataset.caption;
    };

    stack.addEventListener("click", (event) => {
      const card = event.target.closest("[data-photo-card]");
      if (cards.includes(card)) bringForward(card);
    });

    stack.addEventListener("keydown", (event) => {
      if (!event.target.closest("[data-photo-card]")) return;
      const current = cards.indexOf(order[0]);
      let next;
      switch (event.key) {
        case "ArrowRight": next = (current + 1) % cards.length; break;
        case "ArrowLeft": next = (current - 1 + cards.length) % cards.length; break;
        case "Home": next = 0; break;
        case "End": next = cards.length - 1; break;
        default: return;
      }
      event.preventDefault();
      bringForward(cards[next]);
      cards[next].focus({ preventScroll: true });
    });
  });
})();
