const navbarMenu = document.querySelector(".navbar .links");
const hamburgerBtn = document.querySelector(".hamburger-btn");
const hideMenuBtn = navbarMenu.querySelector(".close-btn");
const showPopupBtn = document.querySelector(".login-btn");
const formPopup = document.querySelector(".form-popup");
const hidePopupBtn = formPopup.querySelector(".close-btn");
const signupLoginLink = formPopup.querySelectorAll(".bottom-link a");

hamburgerBtn.addEventListener("click", () => {
  navbarMenu.classList.toggle("show-menu");
});

hideMenuBtn.addEventListener("click", () => hamburgerBtn.click());

showPopupBtn.addEventListener("click", () => {
  document.body.classList.toggle("show-popup");
  document.body.style.transition = "background-image 1s ease-in-out";
  document.body.style.backgroundImage = "url('/static/images/ahero-bg.jpg')";
});

hidePopupBtn.addEventListener("click", () => {
  showPopupBtn.click();
  document.body.style.transition = "background-image 1s ease-in-out";
  document.body.style.backgroundImage = "url('/static/images/hero-bg.jpg')";
});

signupLoginLink.forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    formPopup.classList[link.id === "signup-link" ? "add" : "remove"](
      "show-signup"
    );
  });
});

document.addEventListener("DOMContentLoaded", function () {
  const dropdown = document.querySelector(".dropdown");
  dropdown.addEventListener("click", function () {
    this.classList.toggle("show-menu");
  });
});

