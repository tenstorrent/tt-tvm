# SPDX-FileCopyrightText: © 2019-2023 The Apache Software Foundation
#
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import json
import logging
from urllib import request
from typing import Dict, Any, Optional


def get(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    logging.info(f"Requesting GET to {url}")
    if headers is None:
        headers = {}
    req = request.Request(url, headers=headers)
    with request.urlopen(req) as response:
        response_headers = {k: v for k, v in response.getheaders()}
        response = json.loads(response.read())

    return response, response_headers
