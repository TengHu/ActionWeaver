from typing import List

from pydantic import BaseModel

from actionweaver.actions.factories.function import action


class Place(BaseModel):
    lat: float
    lng: float
    description: str


class Folium:
    def verify_lib_installed(self):
        try:
            import folium
        except ImportError:
            raise ImportError(
                "`folium` package not found, please run `pip install folium`"
            )

    @action(name="ShowMap")
    def show_map(self, places: List[Place]) -> str:
        """
        Display a map with the provided latitude and longitude coordinates.

        This action requires proper integration with map services and may require API keys or authentication.
        Ensure that relevant documentation and dependencies are properly set up.

        This method requires `folium` installed
        """
        self.verify_lib_installed()

        import folium

        # Calculate the center of all places
        avg_lat = sum(place["lat"] for place in places) / len(places)
        avg_lng = sum(place["lng"] for place in places) / len(places)

        # Create a folium Map centered at the average latitude and longitude
        m = folium.Map(location=[avg_lat, avg_lng])

        # Add markers for each place
        for place in places:
            folium.Marker(
                [place["lat"], place["lng"]], tooltip=place["description"]
            ).add_to(m)

        # Display the map
        display(m)
        return "Map has been displayed"
