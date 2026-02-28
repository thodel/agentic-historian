const reveals = document.querySelectorAll('[data-reveal]');
const repoMeta = document.querySelector('meta[name="repo-url"]');
const repoLinks = document.querySelectorAll('[data-repo-link]');

if (repoMeta && repoMeta.content) {
  repoLinks.forEach((link) => {
    link.href = repoMeta.content;
  });
}

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          obs.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );

  reveals.forEach((el) => observer.observe(el));
} else {
  reveals.forEach((el) => el.classList.add('is-visible'));
}
