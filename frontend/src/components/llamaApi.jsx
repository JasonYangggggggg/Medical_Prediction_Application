/********************************************
 * llamaApi.jsx
 ********************************************/

/**
 * - Detect "my list" questions (e.g. "what's in my list?")
 * - Detect "sale" questions (e.g. "what's on sale?", "any deals?")
 * - Detect column-specific questions (e.g. "What's in Processed?")
 * - If none of the above, do the standard embeddings approach.
 */
export async function chatWithLlama(userInput) {
    // -------------------------------------
    // 0) NORMALIZE INPUT
    // -------------------------------------
    const normalizedInput = userInput.toLowerCase();
  
    // -------------------------------------
    // 1) DETECT "MY LIST" INTENT
    // -------------------------------------
    const userListKeywords = [
      "what's in my list",
      "whats in my list",
      "my list",
      "my shopping list",
      "show me my list",
      "show me my shopping list",
      "in my list",
      "in my shopping list",
    ];
    const isAskingForUserList = userListKeywords.some((phrase) =>
      normalizedInput.includes(phrase)
    );
  
    // -------------------------------------
    // 2) DETECT "SALE" QUESTIONS
    // -------------------------------------
    const saleKeywords = [
      "sale",
      "on sale",
      "discount",
      "discounted",
      "cheap",
      "deal",
      "deals",
      "special",
      "specials",
    ];
    const isAskingForSale = saleKeywords.some((phrase) =>
      normalizedInput.includes(phrase)
    );
  
    // -------------------------------------
    // 3) DETECT COLUMN-SPECIFIC QUESTIONS
    //    e.g. "What's in processed?", "Show me fruit."
    // -------------------------------------
    // We'll store whichever column user might be referencing in `askedColumnTitle`.
    // If found, we skip embeddings & show that column only.
    let askedColumnTitle = null;
  
    // -------------------------------------
    // 4) LOAD DATA
    // -------------------------------------
    const userList = JSON.parse(localStorage.getItem("userList")) || [];
    const searchResults = JSON.parse(localStorage.getItem("searchResults")) || [];
    const kanbanColumns = JSON.parse(localStorage.getItem("kanbanColumns")) || [];
  
    // NEW: We'll read the onSale state from localStorage
    const onSale = JSON.parse(localStorage.getItem("onSale")) || [];
  
    // Flatten & label items from kanbanColumns
    const kanbanArray = Object.entries(kanbanColumns).sort((a, b) => {
      const titleA = a[1].title.toLowerCase();
      const titleB = b[1].title.toLowerCase();
      return titleA.localeCompare(titleB);
    });
  
    // A helper to calculate total price for a column’s items
    function calcColumnTotalPrice(items) {
      let total = 0;
      items.forEach((item) => {
        const numericPrice = parseFloat(item.price?.replace(/[^0-9.]/g, "")) || 0;
        total += numericPrice;
      });
      return total.toFixed(2);
    }
  
    const kanbanItems = kanbanArray.flatMap(([key, column]) => {
      const colTitle = column.title || "No Column Title";
      const items = column.items || [];
      // We pre-calc the total price for this column
      const colTotal = calcColumnTotalPrice(items);
  
      return items.map((item) => ({
        ...item,
        columnTitle: colTitle,
        source: "Kanban",
        columnTotal: colTotal,
      }));
    });
  
    // Now gather column titles for user references
    const columnTitlesLower = kanbanArray.map(([key, col]) =>
      col.title.toLowerCase()
    );
  
    // Check if user specifically asked about one of those columns
    for (const colTitleLower of columnTitlesLower) {
      if (normalizedInput.includes(colTitleLower)) {
        askedColumnTitle = colTitleLower;
        break;
      }
    }
  
    // Label userList & searchResults items
    const labeledUserList = userList.map((item) => ({
      ...item,
      columnTitle: "UserList",
      source: "UserList",
    }));
    const labeledSearchResults = searchResults.map((item) => ({
      ...item,
      columnTitle: "RecentSearch",
      source: "SearchResults",
    }));
  
    // Label onSale items as belonging to a SALE section
    const labeledOnSale = onSale.map((item) => ({
      ...item,
      columnTitle: "SALE",
      source: "OnSale",
    }));
  
    // Combine them all
    const allItems = [
      ...labeledUserList,
      ...labeledSearchResults,
      ...labeledOnSale,
      ...kanbanItems,
    ];
  
    // Helper that turns an item into text for embedding
    function itemToSnippet(item) {
      // If it's from the onSale array or has sale_price/was_price, we consider it "Yes"
      const hasSaleIndicator =
        item.source === "OnSale" ||
        Boolean(item.sale_price) ||
        Boolean(item.was_price);
      const onSaleLabel = hasSaleIndicator ? "Yes" : "No";
  
      const maybeColTotal = item.columnTotal
        ? ` (Column total price: $${item.columnTotal})`
        : "";
  
      return [
        `ITEM (source: ${item.source ?? "(unknown)"}) :`,
        `  category: ${item.columnTitle ?? "(none)"}${maybeColTotal}`,
        `  title: ${item.title ?? "(none)"}`,
        `  brand: ${item.brand ?? "(none)"}`,
        `  package_size: ${item.package_size ?? "(none)"}`,
        `  price: ${item.price ?? "(none)"}`,
        `  sale_price: ${item.sale_price ?? "(none)"}`,
        `  was_price: ${item.was_price ?? "(none)"}`,
        `  onSale: ${onSaleLabel}`,
        `  address: ${item.store_address ?? "(none)"}`,
        `  link: ${item.product_link ?? "(none)"}`,
      ].join("\n");
    }
  
    // -------------------------------------
    // 5) IF USER IS ASKING FOR "MY LIST"
    // -------------------------------------
    if (isAskingForUserList) {
      const userListSnippets = labeledUserList.map(itemToSnippet);
      const contextText = userListSnippets
        .map((snippet, i) => `#### UserList Item #${i + 1}\n${snippet}`)
        .join("\n\n");
  
      // Build payload
      const chatPayload = {
        model: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        messages: [
          {
            role: "system",
            content: `The user wants to know what's in their shopping list (userList).
  Here are the items:
  
  ===== START OF USERLIST DATA =====
  ${contextText}
  ===== END OF USERLIST DATA =====
  `,
          },
          {
            role: "user",
            content: userInput,
          },
        ],
        temperature: 0.7,
        stream: false,
      };
  
      console.log("===== MY LIST Shortcut =====");
      return await callChatEndpoint(chatPayload);
    }
  
    // -------------------------------------
    // 6) IF USER IS ASKING FOR "SALE" ITEMS
    // -------------------------------------
    if (isAskingForSale) {
      // We'll gather items from the onSale array (already labeled with "SALE")
      const saleSnippets = labeledOnSale.map(itemToSnippet);
  
      const contextText = saleSnippets
        .map((snippet, i) => `#### OnSale Item #${i + 1}\n${snippet}`)
        .join("\n\n");
  
      const chatPayload = {
        model: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        messages: [
          {
            role: "system",
            content: `The user wants to see items on sale/discount/deal. 
  Here are the "SALE" items from onSale state:
  
  ===== START OF SALE DATA =====
  ${contextText}
  ===== END OF SALE DATA =====
  `,
          },
          {
            role: "user",
            content: userInput,
          },
        ],
        temperature: 0.7,
        stream: false,
      };
  
      console.log("===== SALE Shortcut =====");
      return await callChatEndpoint(chatPayload);
    }
  
    // -------------------------------------
    // 7) IF USER ASKED ABOUT A KNOWN COLUMN
    // -------------------------------------
    if (askedColumnTitle) {
      // Filter out the items from that specific column
      const matchedItems = kanbanItems.filter(
        (it) => it.columnTitle.toLowerCase() === askedColumnTitle
      );
  
      const columnSnippets = matchedItems.map(itemToSnippet);
  
      const colTotal = matchedItems.length
        ? matchedItems[0].columnTotal
        : "0.00";
  
      const contextText = columnSnippets
        .map(
          (snippet, i) =>
            `#### Column "${askedColumnTitle}" Item #${i + 1}\n${snippet}`
        )
        .join("\n\n");
  
      const chatPayload = {
        model: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        messages: [
          {
            role: "system",
            content: `User is asking about the "${askedColumnTitle}" column. 
  This column has a total price of $${colTotal}. 
  
  ===== START OF COLUMN DATA =====
  ${contextText}
  ===== END OF COLUMN DATA =====
  `,
          },
          {
            role: "user",
            content: userInput,
          },
        ],
        temperature: 0.7,
        stream: false,
      };
  
      console.log("===== COLUMN Shortcut =====");
      return await callChatEndpoint(chatPayload);
    }
  
    // -------------------------------------
    // 8) OTHERWISE: DO NORMAL EMBEDDING APPROACH
    // -------------------------------------
    const itemsWithEmbeddings = [];
    for (const item of allItems) {
      const snippet = itemToSnippet(item);
      const embedding = await getEmbedding(snippet);
      itemsWithEmbeddings.push({
        item,
        snippet,
        embedding,
      });
    }
  
    // (B) Embed the user query
    const userQueryEmbedding = await getEmbedding(userInput);
  
    // (C) Compute similarities
    const ranked = itemsWithEmbeddings
      .map((obj) => {
        const sim = cosineSimilarity(userQueryEmbedding, obj.embedding);
        return { ...obj, similarity: sim };
      })
      .sort((a, b) => b.similarity - a.similarity);
  
    // Keep top 3 or so
    const topN = ranked.slice(0, 3);
  
    // Build context text for topN
    const contextText = topN
      .map(
        (obj, i) =>
          `#### Relevant Item #${i + 1} (similarity: ${obj.similarity.toFixed(3)})\n${obj.snippet}`
      )
      .join("\n\n");
  
    // (D) Final system prompt
    const chatPayload = {
      model: "bartowski/Llama-3.2-3B-Instruct-GGUF",
      messages: [
        {
          role: "system",
          content: `You have access to the user's shopping data, including:
  - userList = user's current shopping list
  - searchResults = user's most recent search
  - onSale = items flagged as on sale
  - kanbanColumns = a board with columns like "Default List," "Processed," "Liquids," "Fruit," etc.
  
  Use ONLY the items below to answer questions. If the info is not in these items, say "I don't know."
  
  ===== START OF RELEVANT DATA =====
  ${contextText}
  ===== END OF RELEVANT DATA =====
  `,
        },
        {
          role: "user",
          content: userInput,
        },
      ],
      temperature: 0.7,
      stream: false,
    };
  
    console.log("===== EMBEDDING PATH =====");
    return await callChatEndpoint(chatPayload);
  }
  
  /********************************************
   * callChatEndpoint: a helper to avoid duplication
   ********************************************/
  async function callChatEndpoint(chatPayload) {
    try {
      const response = await fetch("http://26.81.189.101:1234/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Basic " + btoa("api:lm-studio"),
        },
        body: JSON.stringify(chatPayload),
      });
  
      if (!response.ok) {
        console.error("Error in response:", response.status, response.statusText);
        return "Something went wrong.";
      }
  
      const data = await response.json();
      return data?.choices?.[0]?.message?.content || "No response";
    } catch (error) {
      console.error("Error:", error);
      return "Error occurred while fetching response.";
    }
  }
  
  /********************************************
   * getEmbedding: calls Llama /v1/embeddings
   ********************************************/
  async function getEmbedding(text) {
    const payload = {
      model: "bartowski/Llama-3.2-3B-Instruct-GGUF",
      input: text,
    };
  
    try {
      const response = await fetch("http://26.81.189.101:1234/v1/embeddings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Basic " + btoa("api:lm-studio"),
        },
        body: JSON.stringify(payload),
      });
  
      if (!response.ok) {
        console.error("Embedding error:", response.status, response.statusText);
        return [];
      }
  
      const data = await response.json();
      return data?.data?.[0]?.embedding || [];
    } catch (error) {
      console.error("Embedding error:", error);
      return [];
    }
  }
  
  /********************************************
   * cosineSimilarity: simple helper
   ********************************************/
  function cosineSimilarity(vecA, vecB) {
    if (!Array.isArray(vecA) || !Array.isArray(vecB) || vecA.length !== vecB.length) {
      return 0;
    }
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) {
      return 0;
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }
  