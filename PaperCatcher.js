let dic = null

function initialize(data){
    dic = data


    let container_navi = document.getElementById('navi');
    fetch('data/navi_abbr.json')
        .then(response => response.json())
        .then(data => navigation_list(data, container_navi))
        .catch(error => console.error('Error in navi:', error));

    let container_latest = document.getElementById('latest');
    fetch('data/latest.json')
    		.then(response => response.json())
    		.then(data => latestPapers(data, container_latest))
    		.catch(error => console.error('Error in latest:', error));

    let container_main = document.getElementById('contents');
    let paper_counter = document.getElementById('counter');
    fetch('data/articles.json')
        .then(response => response.json())
        .then(data => mainPapers(data, container_main, paper_counter))
        .catch(error => console.error('Error in main:', error));

    let container_ac = document.getElementById('thanks');
    fetch('data/acknowledgements.json')
        .then(response => response.json())
        .then(data => acknowledgment(data, container_ac))
        .catch(error => console.error('Error in acknowledgements:', error));
}

function mainPapers(data, container, counter=null){
    let container_navi = document.getElementById('navi');
    let debug_block = document.getElementById('debug');

    let n_paper = 0;
    Object.keys(data).forEach(field=>{
        let ul_field = document.createElement('ul');
        ul_field.innerHTML += `<li><h3 class="section-title headline-medium on-background-text" id="${field}">${dic[field]}</h3></li>`;

        let div_navi = document.createElement('div');
        div_navi.setAttribute('class', 'list-section primary-container');
        div_navi.innerHTML += `<div class="navigation-item"><a onclick="close_sidebar()" href="#${field}" class="title-large on-primary-container-text">${dic[field]}</a></div>`;

        let fields = data[field];
        Object.keys(fields).forEach(section=>{
            let li_section = document.createElement('li');
            if (section.includes("_")){
                if(dic[section]){
                    li_section.innerHTML = `<h5 class="section-subtitle title-large on-background-text" id=${section}>${dic[section]}</h5>`;
                }
            }else{
                if(dic[section]){
                    li_section.innerHTML = `<h4 class="section-subtitle title-large on-background-text" id=${section}>${dic[section]}</h4>`;
                    div_navi.innerHTML += `<div class="navigation-item"><a onclick="close_sidebar()" href="#${section}" class="title-medium on-primary-container-text">${dic[section]}</a></div>`;
                }
            }
            ul_field.appendChild(li_section)
            if (section === "graph-neural"){
                let li_comment =  document.createElement('li');
                li_comment.setAttribute('class', "essay-container title-large error-container on-error-container-text");
                li_comment.innerHTML = `<div><p class="essay-content">No further updates will be made regarding graph distillation topics as sufficient papers and summary projects are already available on the subject</p></div>`;
                ul_field.appendChild(li_comment);
            }
            let n_paper_f = 0;
            let papers = fields[section];
            papers.forEach(paper =>{
                aPaper(paper, ul_field);
                n_paper += 1;
                n_paper_f += 1;
            });
            // debug_block.innerHTML += `++${n_paper_f}`;
        });
        container.appendChild(ul_field);
        container_navi.appendChild(div_navi);
    })
    if (counter){
        counter.innerHTML = n_paper;
    }
}

function navigation_list(data, container){
    // Object.keys(data).forEach(field=>{
    //     let div = document.createElement('div');
    //     div.setAttribute('class', 'list-section primary-container');
    //     div.innerHTML += `<div class="navigation-item"><a onclick="close_sidebar()" href="#${}" class="title-large on-primary-container-text">Main</a></div>`;
    //     container.appendChild(div);
    // });
}

function latestPapers(data, container){
    let ul_field = document.createElement('ul');
    Object.keys(data).forEach(date=>{
        ul_field.innerHTML += `<li class="edit-date">
                                <p class="essay-content title-large on-surface-variant-text">${date}</p></li>`;
        ul_field.innerHTML += `<li><hr class="outline-variant" style="width:96%;text-align:center;"></li>`;
        let papers = data[date];
        papers.forEach(paper=>{
            aPaper(paper, ul_field);
        })
    });
    container.appendChild(ul_field);
}

function aPaper(paper, container){
    let li_paper = document.createElement('li');
    li_paper.setAttribute('class', "essay-container title-large surface-variant on-surface-variant-text");

    let div_content = document.createElement('div');
    if (paper['github']){
        div_content.innerHTML += `<p class="essay-content">
                                  <a class="on-surface-variant-text" href=${paper.github}>${paper.title}</a></p>`;
    }else{
        div_content.innerHTML += `<p class="essay-content">${paper.title}</p>`;
    }
    div_content.innerHTML += `<p class="title-medium essay-author"><i>${paper.author}</i></p>`;
    li_paper.appendChild(div_content);
    container.appendChild(li_paper);

    let div_btn = document.createElement('div');
    div_btn.setAttribute('class', "button-group");
    // 用placeholder达成全部靠右排列的效果
    let n_space = 0;
    let btn_html = "";
    if (paper['website']){
        btn_html += `<a href=${paper.website}>
                            <span class="essay-button surface">
                            <svg class="icon" xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24">
                            <path d="M480-80q-82 0-155-31.5t-127.5-86Q143-252 111.5-325T80-480q0-83 31.5-155.5t86-127Q252-817 325-848.5T480-880q83 0 155.5 31.5t127 86q54.5 54.5 86 127T880-480q0 82-31.5 155t-86 127.5q-54.5 54.5-127 86T480-80Zm0-82q26-36 45-75t31-83H404q12 44 31 83t45 75Zm-104-16q-18-33-31.5-68.5T322-320H204q29 50 72.5 87t99.5 55Zm208 0q56-18 99.5-55t72.5-87H638q-9 38-22.5 73.5T584-178ZM170-400h136q-3-20-4.5-39.5T300-480q0-21 1.5-40.5T306-560H170q-5 20-7.5 39.5T160-480q0 21 2.5 40.5T170-400Zm216 0h188q3-20 4.5-39.5T580-480q0-21-1.5-40.5T574-560H386q-3 20-4.5 39.5T380-480q0 21 1.5 40.5T386-400Zm268 0h136q5-20 7.5-39.5T800-480q0-21-2.5-40.5T790-560H654q3 20 4.5 39.5T660-480q0 21-1.5 40.5T654-400Zm-16-240h118q-29-50-72.5-87T584-782q18 33 31.5 68.5T638-640Zm-234 0h152q-12-44-31-83t-45-75q-26 36-45 75t-31 83Zm-200 0h118q9-38 22.5-73.5T376-782q-56 18-99.5 55T204-640Z"/>
                            </svg></span></a>`;
    }else {
        n_space += 1;
    }
    if (paper['cite']){
        btn_html += `<a href=${paper.cite}>
                            <span class="essay-button surface">
                            <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 0 24 24" width="24px">
                            <path d="M0 0h24v24H0V0z" fill="none"/><path d="M18.62 18h-5.24l2-4H13V6h8v7.24L18.62 18zm-2-2h.76L19 12.76V8h-4v4h3.62l-2 4zm-8 2H3.38l2-4H3V6h8v7.24L8.62 18zm-2-2h.76L9 12.76V8H5v4h3.62l-2 4z"/>
                            </svg></span></a>`;
    }else{
        n_space += 1;
    }
    if (paper['url']){
        btn_html += `<a href=${paper.url}>
                            <span class="essay-button surface">
                            <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24">
                            <path d="m720-120 160-160-56-56-64 64v-167h-80v167l-64-64-56 56 160 160ZM560 0v-80h320V0H560ZM240-160q-33 0-56.5-23.5T160-240v-560q0-33 23.5-56.5T240-880h280l240 240v121h-80v-81H480v-200H240v560h240v80H240Zm0-80v-560 560Z"/>
                            </svg></span></a>`;
    }else{
        n_space += 1;
    }
    for (let i = 0; i < n_space; i++) {
        div_btn.innerHTML += `<span class="placeholder"></span>`;
    }
    div_btn.innerHTML += btn_html;

    li_paper.appendChild(div_btn);
    container.appendChild(li_paper);
}

function acknowledgment(data, container){
    container.innerHTML += `<h4 class="page-footer-subtitle body-medium on-background-text">Acknowledgments</h4>`;
    let thanks_html = `<p class="page-footer-instruction body-large on-background-text">We would like to express our heartfelt thanks to `;
    let names = Object.keys(data);
    let n = names.length;
    for (let i= 0; i < n; i ++){
        name = names[i];
        if (i === (n-1)){
            thanks_html += `and <a class="primary-text" href="${data[name]}">${name}</a> `;
        }else{
            thanks_html += `<a class="primary-text" href="${data[name]}">${name}</a>, `;
        }
    }
    thanks_html += `for their valuable suggestions and contributions.</p>`;
    container.innerHTML += thanks_html;

    container.innerHTML += `<p class="page-footer-instruction body-large on-background-text">
                            The <a class="primary-text" href="https://guang000.github.io/Awesome-Dataset-Distillation/">Homepage </a>
                            of Awesome Dataset Distillation was designed by 
                            <a class="primary-text" href="https://lovelessg.github.io/">Longzhen Li </a>
                            and maintained by <a class="primary-text" href="https://github.com/SumomoTaku">Mingzhuo Li</a></p>`
}

// actions
function open_sidebar() {
    let isMobile = navigator.userAgent.toLowerCase().match(/mobile/i)
    if (isMobile) {
        document.getElementById("quickContent").style.width = "100%";
        document.getElementById("sidebarTitle").style.width = "100%";
    } else {
        document.getElementById("quickContent").style.width = "466px";
        document.getElementById("sidebarTitle").style.width = "466px";
        document.getElementById("sidebarOverlay").style.width = "100%";
        document.getElementById("sidebarOverlay").style.height = "100%";
        document.getElementById("awesome-dataset-distillation").style.overflow = "hidden";
    }

    document.getElementById("sidebarTitle").style.position = "fixed";
}

function close_sidebar() {
    document.getElementById("quickContent").style.width = "0";
    document.getElementById("sidebarTitle").style.position = "static";
    document.getElementById("sidebarOverlay").style.width = "0";
    document.getElementById("sidebarOverlay").style.height = "0";
    document.getElementById("awesome-dataset-distillation").style.overflow = "visible";
}

function github_star(){
    fetch('https://api.github.com/repos/guang000/awesome-dataset-distillation')
        .then(response => {response.json()})
        .then(data => {
            document.getElementById("starCount").innerHTML = data.stargazers_count;
        })
}