// Enhanced JavaScript for LLMs GitHub Pages site

document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add fade-in animation to navigation cards
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe navigation cards and content sections
    const animatedElements = document.querySelectorAll('.nav-card, .content-section, table, pre');
    animatedElements.forEach(el => {
        observer.observe(el);
    });

    // Add copy button to code blocks
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(codeBlock => {
        const pre = codeBlock.parentElement;
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = '📋 Copy';
        copyButton.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
        `;
        
        pre.style.position = 'relative';
        pre.appendChild(copyButton);
        
        copyButton.addEventListener('click', function() {
            navigator.clipboard.writeText(codeBlock.textContent).then(() => {
                copyButton.innerHTML = '✅ Copied!';
                copyButton.style.background = '#10b981';
                copyButton.style.color = 'white';
                setTimeout(() => {
                    copyButton.innerHTML = '📋 Copy';
                    copyButton.style.background = 'rgba(255, 255, 255, 0.9)';
                    copyButton.style.color = 'inherit';
                }, 2000);
            });
        });
        
        copyButton.addEventListener('mouseenter', function() {
            this.style.background = '#f3f4f6';
        });
        
        copyButton.addEventListener('mouseleave', function() {
            if (this.innerHTML === '📋 Copy') {
                this.style.background = 'rgba(255, 255, 255, 0.9)';
            }
        });
    });

    // Add table of contents generator
    function generateTableOfContents() {
        const headings = document.querySelectorAll('h2, h3');
        if (headings.length === 0) return;

        const tocContainer = document.createElement('div');
        tocContainer.className = 'table-of-contents';
        tocContainer.innerHTML = '<h3>Table of Contents</h3>';
        
        const tocList = document.createElement('ul');
        tocList.style.cssText = `
            list-style: none;
            padding-left: 0;
            margin: 1rem 0;
        `;

        headings.forEach((heading, index) => {
            // Create ID if it doesn't exist
            if (!heading.id) {
                heading.id = heading.textContent.toLowerCase()
                    .replace(/[^\w\s-]/g, '')
                    .replace(/\s+/g, '-')
                    .replace(/--+/g, '-')
                    .trim();
            }

            const listItem = document.createElement('li');
            const link = document.createElement('a');
            link.href = `#${heading.id}`;
            link.textContent = heading.textContent;
            link.style.cssText = `
                display: block;
                padding: 0.25rem 0;
                color: #6b7280;
                text-decoration: none;
                transition: color 0.3s ease;
                ${heading.tagName === 'H3' ? 'padding-left: 1rem; font-size: 0.9rem;' : ''}
            `;
            
            link.addEventListener('mouseenter', function() {
                this.style.color = '#2563eb';
            });
            
            link.addEventListener('mouseleave', function() {
                this.style.color = '#6b7280';
            });

            listItem.appendChild(link);
            tocList.appendChild(listItem);
        });

        tocContainer.appendChild(tocList);
        tocContainer.style.cssText = `
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 2rem 0;
            position: sticky;
            top: 2rem;
            max-height: 70vh;
            overflow-y: auto;
        `;

        // Insert TOC after the first paragraph
        const firstParagraph = document.querySelector('p');
        if (firstParagraph) {
            firstParagraph.parentNode.insertBefore(tocContainer, firstParagraph.nextSibling);
        }
    }

    // Generate TOC if there are enough headings
    const headingCount = document.querySelectorAll('h2, h3').length;
    if (headingCount >= 5) {
        generateTableOfContents();
    }

    // Add progress indicator
    function createProgressIndicator() {
        const progressBar = document.createElement('div');
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            z-index: 9999;
            transition: width 0.3s ease;
        `;
        document.body.appendChild(progressBar);

        window.addEventListener('scroll', function() {
            const scrollTop = window.pageYOffset;
            const docHeight = document.body.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;
            progressBar.style.width = scrollPercent + '%';
        });
    }

    createProgressIndicator();

    // Add back to top button
    function createBackToTopButton() {
        const backToTop = document.createElement('button');
        backToTop.innerHTML = '↑';
        backToTop.style.cssText = `
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: #2563eb;
            color: white;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        `;

        document.body.appendChild(backToTop);

        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                backToTop.style.opacity = '1';
                backToTop.style.visibility = 'visible';
            } else {
                backToTop.style.opacity = '0';
                backToTop.style.visibility = 'hidden';
            }
        });

        backToTop.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });

        backToTop.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.1)';
            this.style.background = '#1e40af';
        });

        backToTop.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.background = '#2563eb';
        });
    }

    createBackToTopButton();

    // Add search functionality
    function createSearchBox() {
        const searchContainer = document.createElement('div');
        searchContainer.style.cssText = `
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 1000;
        `;

        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.placeholder = 'Search content...';
        searchInput.style.cssText = `
            padding: 0.5rem 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background: white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            width: 200px;
            font-size: 0.875rem;
        `;

        searchContainer.appendChild(searchInput);
        document.body.appendChild(searchContainer);

        let searchTimeout;
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            const query = this.value.toLowerCase();
            
            searchTimeout = setTimeout(() => {
                const textElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, td');
                
                textElements.forEach(element => {
                    if (element.style) {
                        element.style.backgroundColor = '';
                    }
                });

                if (query.length > 2) {
                    textElements.forEach(element => {
                        if (element.textContent.toLowerCase().includes(query)) {
                            element.style.backgroundColor = '#fef3c7';
                            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        }
                    });
                }
            }, 300);
        });
    }

    // Only add search on larger screens
    if (window.innerWidth > 768) {
        createSearchBox();
    }

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K for search focus
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('input[placeholder="Search content..."]');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to clear search
        if (e.key === 'Escape') {
            const searchInput = document.querySelector('input[placeholder="Search content..."]');
            if (searchInput) {
                searchInput.value = '';
                searchInput.blur();
                // Clear highlights
                const textElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, td');
                textElements.forEach(element => {
                    if (element.style) {
                        element.style.backgroundColor = '';
                    }
                });
            }
        }
    });

    console.log('🚀 LLMs GitHub Pages site enhanced with interactive features!');
});
