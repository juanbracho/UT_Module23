document.getElementById("tickerForm").addEventListener("submit", function (e) {
    e.preventDefault(); // Prevent default form submission

    // Show the spinner
    const spinner = document.getElementById("loadingSpinner");
    if (spinner) {
        spinner.style.display = "flex";
    }

    // Wait at least 500ms before submitting the form
    setTimeout(() => {
        e.target.submit(); // Proceed with form submission
    }, 500);
});
