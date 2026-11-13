---
layout: editorial
modern_layout: true
title: "Site map"
kicker: "Explore / Navigation"
description: "Find your way around my site."
permalink: /sitemap/
author_profile: false
sitemap: false
---

<ul class="site-map-list">
  {% for entry in site.data.site-map %}
  <li><h2><a href="{{ entry.url | relative_url }}">{{ entry.title | escape }}</a></h2><p>{{ entry.description | escape }}</p></li>
  {% endfor %}
  <li><h2><a href="{{ site.author.blog }}">Blog <span aria-hidden="true">↗</span></a></h2><p>My essays and learning notes on Notion.</p></li>
</ul>
