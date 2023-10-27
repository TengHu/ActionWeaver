from actionweaver import action


class LangChainTools:
    def verify_lib_installed(self):
        try:
            import langchain
        except ImportError:
            raise ImportError(
                "`langchain` package not found, please run `pip install langchain`"
            )

    @action(name="GoogleSearch")
    def search(self, query: str):
        """
        Perform a Google search and return query results with titles and links.

        :param query: The search query to be used for the Google search.
        """
        from langchain.utilities import GoogleSearchAPIWrapper

        search = GoogleSearchAPIWrapper()
        res = search.results(query, 10)
        formatted_data = ""

        # Iterate through the data and append each item to the formatted_data string
        for idx, item in enumerate(res):
            formatted_data += f"({idx}) {item['title']}: {item['snippet']}\n"
            formatted_data += f"[Source]: {item['link']}\n\n"

        return formatted_data
