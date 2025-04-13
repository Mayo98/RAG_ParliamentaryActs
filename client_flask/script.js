const SERVER_URL = "..."; //  IP 
document.addEventListener("DOMContentLoaded", function () {
    const optionSelect = document.getElementById("optionSelect");
    const queryInput = document.getElementById("queryInput");
    const partySelect = document.createElement("select");
    const lawTypeSelect = document.createElement("select");
    const lawInput = document.createElement("input");

    // Creazione dell'input con ricerca per i deputati
    const deputyInput = document.createElement("input");
    deputyInput.style.width = "100%";
    deputyInput.style.border = "2px solid red";
    deputyInput.style.maxWidth = "400px";
    deputyInput.style.margin = "10px 0";
    deputyInput.style.padding = "10px";
    deputyInput.style.fontSize = "16px";
    deputyInput.style.border = "1px solid #ccc";
    deputyInput.style.borderRadius = "5px";
    deputyInput.setAttribute("type", "text");
    deputyInput.setAttribute("id", "deputyInput");
    deputyInput.setAttribute("placeholder", "Cerca un deputato...");
    
    const deputyDatalist = document.createElement("datalist");
    deputyDatalist.setAttribute("id", "deputyDatalist");
    deputyInput.setAttribute("list", "deputyDatalist"); // Associa l'input alla datalist

    queryInput.addEventListener("queryInput", function () {
        this.style.height = "auto"; 
        this.style.height = (this.scrollHeight) + "px"; 
    });

    // Configura il campo di testo per il riassunto della proposta di legge
    lawInput.setAttribute("type", "text");
    lawInput.setAttribute("id", "lawInput");
    lawInput.setAttribute("placeholder", "Inserisci il link della proposta di legge");

    // Configura la listbox per il tipo di riassunto
    lawTypeSelect.setAttribute("id", "lawTypeSelect");
    const summaryOption = new Option("Riassunto", "summary");
    const detailedSummaryOption = new Option("Riassunto Dettagliato", "detailed_summary");
    const queryLawOption = new Option("Chiedi qualcosa su questo atto");
    lawTypeSelect.appendChild(summaryOption);
    lawTypeSelect.appendChild(detailedSummaryOption);
    lawTypeSelect.appendChild(queryLawOption);

    // Contenitore dinamico per gli elementi extra
    const extraInputContainer = document.createElement("div");
    extraInputContainer.setAttribute("id", "extraInputContainer");

    // Inserisci il container prima dell'input principale
    queryInput.parentNode.insertBefore(extraInputContainer, queryInput);

    // Aggiorna l'interfaccia quando cambia la selezione
    optionSelect.addEventListener("change", function () {
        extraInputContainer.innerHTML = ""; // Pulisce il contenuto precedente

        if (optionSelect.value === "Ricerca opinione di un deputato") {
            extraInputContainer.appendChild(deputyInput);
            extraInputContainer.appendChild(deputyDatalist);
            fetchDeputies(); // Carica i deputati dal server
        } else if (optionSelect.value === "Opinioni di un partito politico") {
            extraInputContainer.appendChild(partySelect);
            fetchParties(); // Carica i partiti dal server
        } else if (optionSelect.value === "Informazioni su una proposta di legge") {
            extraInputContainer.appendChild(lawInput);
            extraInputContainer.appendChild(lawTypeSelect);
        }
    });

    // Funzione per caricare i deputati e popolare la datalist
    function fetchDeputies() {
        fetch(SERVER_URL + "/get_deputati")
            .then(response => response.json())
            .then(data => {
                console.log("Dati ricevuti dal server:", data);
                deputyDatalist.innerHTML = ""; // Pulisce il datalist

                data.forEach(deputato => {
                    let option = document.createElement("option");
                    option.value = deputato; // Imposta il valore dell'opzione
                    deputyDatalist.appendChild(option);
                });
            })
            .catch(error => console.error("Errore nel caricamento deputati:", error));
    }

    // Funzione per caricare i partiti 
    function fetchParties() {
        fetch(SERVER_URL + "/get_gruppi")
            .then(response => response.json())
            .then(data => {
                partySelect.innerHTML = '<option value="">Seleziona un partito</option>';
                data.forEach(partito => {
                    let option = document.createElement("option");
                    option.value = partito;
                    option.textContent = partito;
                    partySelect.appendChild(option);
                });
            })
            .catch(error => console.error("Errore nel caricamento partiti:", error));
    }

    // Funzione per chiamata API Flask per LLM 
    window.sendQuery = function () {
        const selectedOption = optionSelect.value;
        const userQuery = queryInput.value;
        let filter = "";
        let optionKey = ""
        let summaryType = "";

        if (selectedOption === "Ricerca opinione di un deputato") {
            filter = deputyInput.value;  // Usa il valore dell'input testuale con ricerca
            optionKey = "deputato";
        } else if (selectedOption === "Opinioni di un partito politico") {
            filter = partySelect.value;
            optionKey = "gruppo";
        } else if (selectedOption === "Informazioni su una proposta di legge") {
            filter = lawInput.value.replace("https://", "http://");  
            optionKey = "uriLegge";
            summaryType = lawTypeSelect.value;
        }

        console.log("Filtro selezionato:", filter);
        
        // Controllo se i campi sono compilati
        if (!userQuery && !(summaryType === "summary" || summaryType === "detailed_summary")) {
            alert("Inserisci una domanda prima di cercare.");
            return;
        }
    
        if ((selectedOption === "Ricerca opinione di un deputato" && !filter) ||
            (selectedOption === "Opinioni di un partito politico" && !filter) ||
            (selectedOption === "Contenuto di una proposta di legge" && (!filter || !summaryType))) {
            alert("Seleziona un valore dalla lista.");
            return;
        }
    
        const requestData = {
            optionKey: optionKey,
            query: userQuery || "",
            filterValue: filter
        };

        fetch(SERVER_URL + "/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData)
        })
            .then(response => response.json())
            .then(data => {
                document.getElementById("results").innerHTML = `<p><strong>Risultato:</strong> ${data.result}</p>`;
            })
            .catch(error => console.error("Errore nella ricerca:", error));
    };
});
