// Handle the form submission with a spinner for better UX
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

// Fetch the tickers from the API and populate the dropdown
document.addEventListener("DOMContentLoaded", () => {
    fetch("/api/tickers")
        .then(response => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            return response.json();
        })
        .then(data => {
            console.log("Tickers received:", data); // Debugging log
            const dropdown = document.getElementById("ticker");
            dropdown.innerHTML = ""; // Clear any existing options

            // Populate the dropdown with fetched tickers
            data.forEach(item => {
                const option = document.createElement("option");
                option.value = item.symbol; // Use the symbol as the value
                option.textContent = `${item.symbol} - ${item.company}`; // Display both symbol and company
                dropdown.appendChild(option);
            });
        })
        .catch(error => console.error("Error fetching tickers:", error)); // Log any errors
});