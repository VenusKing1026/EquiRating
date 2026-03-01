chrome.action.onClicked.addListener((tab) => {
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => document.body.outerHTML
    }, (results) => {

        if (!results || !results[0]) {
            console.error("No body returned");
            return;
        }

        const bodyHTML = results[0].result;
        // 生成合法的文件名（只保留字母数字和下划线）
        let filename = tab.url.split("/").filter(Boolean).pop() || "page";
        filename = filename.replace(/[^a-zA-Z0-9_\-\.]/g, "_") + ".html";
        // 使用 data:URL 方式下载
        const url = "data:text/html;charset=utf-8," + encodeURIComponent(bodyHTML);
        chrome.downloads.download({
            url: url,
            filename: filename,
            saveAs: false
        });
    });
});