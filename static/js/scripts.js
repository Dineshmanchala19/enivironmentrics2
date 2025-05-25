document.addEventListener('DOMContentLoaded', function() {
 

  // Example of interactivity: show an alert when clicking the 'About' button
  const aboutButton = document.querySelector('a[href="/about"]');
  const loginButton = document.querySelector('a[href="/login"]');
  const signupButton = document.querySelector('a[href="/signup"]');
  const ndviButton = document.querySelector('a[href="/ndvi"]');
  const ndwiButton = document.querySelector('a[href="/ndwi"]');
  const nsmiButton = document.querySelector('a[href="/nsmi"]');
  const nsmiButton = document.querySelector('a[href="/dashboard"]');

  if (aboutButton) {
      aboutButton.addEventListener('click', function(event) {
          alert("You are navigating to the About page!");
      });
  }

  if (loginButton) {
      loginButton.addEventListener('click', function(event) {
          alert("You are navigating to the Login page!");
      });
  }

  if (signupButton) {
      signupButton.addEventListener('click', function(event) {
          alert("You are navigating to the Signup page!");
      });
  }

  if (ndviButton) {
      ndviButton.addEventListener('click', function(event) {
          alert("You are navigating to the NDVI page!");
      });
  }

  if (ndwiButton) {
      ndwiButton.addEventListener('click', function(event) {
          alert("You are navigating to the NDWI page!");
      });
  }

  if (nsmiButton) {
      nsmiButton.addEventListener('click', function(event) {
          alert("You are navigating to the NSMI page!");
      });
  }

});
