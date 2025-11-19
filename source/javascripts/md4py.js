// hide the first line in TOC of generated HTML for each module
document.querySelectorAll('a[href^="#onehealth_db."]').forEach(el => {
    // get the part after "#onehealth_db."
    const suffix = el.getAttribute('href').slice('#onehealth_db.'.length);
    // hide only if suffix does not contain a dot
    if (!suffix.includes('.')) {
        el.style.display = 'none';
    }
});