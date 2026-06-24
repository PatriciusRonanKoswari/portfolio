(function () {
    document.addEventListener("DOMContentLoaded", function () {
        initScrollProgress();
    });

    function initScrollProgress() {
        const progress = document.querySelector(".scroll-progress");

        if (!progress) {
            return;
        }

        function updateProgress() {
            const scrollTop = window.scrollY || document.documentElement.scrollTop;
            const scrollableHeight = document.documentElement.scrollHeight - window.innerHeight;
            const width = scrollableHeight > 0 ? (scrollTop / scrollableHeight) * 100 : 0;

            progress.style.width = width + "%";
        }

        updateProgress();
        window.addEventListener("scroll", updateProgress, { passive: true });
        window.addEventListener("resize", updateProgress);
    }
})();
