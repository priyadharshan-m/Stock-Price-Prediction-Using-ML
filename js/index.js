function openSideNavBar() {
    const navbar = document.getElementsByClassName("side-nav-bar")[0]; // Access the first element in the collection
    if (navbar.style.width == "0px") {
        navbar.style.width = "50%";
        navbar.style.display = "block";
    } else {
        navbar.style.width = "0px";
        navbar.style.display = "none";
    }
}

function closeSideNavBar() {
    const navbar = document.getElementsByClassName("side-nav-bar")[0]; // Access the first element in the collection
    navbar.style.width = "0px";
    navbar.style.display = "none";
}
