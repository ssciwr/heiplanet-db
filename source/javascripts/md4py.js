// hide the first line in TOC of generated HTML for each module
document.querySelectorAll('a[href^="#heiplanet_db."]').forEach(el => {
    // get the part after "#heiplanet_db."
    const suffix = el.getAttribute('href').slice('#heiplanet_db.'.length);
    // hide only if suffix does not contain a dot
    if (!suffix.includes('.')) {
        el.style.display = 'none';
    }
});